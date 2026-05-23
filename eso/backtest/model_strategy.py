"""Linear model strategy: combine all 9 causal features into one signal.

Trains a Ridge regression on the train half of the feature vector to predict
future log-returns at a chosen horizon, then uses the standardised prediction
as the position. A dead-band suppresses micro-rebalancing.

Honest constraints:
    - The regression is fit on a strictly past slice (first `train_frac` of
      the input). Predictions on the test slice never see their labels.
    - Standardisation parameters (μ, σ of the prediction stream) are computed
      on the train slice and applied to test.
    - The feature vector itself was already produced causally by
      CausalCyclePhase or CausalFeatureBuilder — no UMAP leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_FEATURES = [
    "sin_theta_6h", "cos_theta_6h",
    "sin_theta_24h", "cos_theta_24h",
    "sin_theta_72h", "cos_theta_72h",
    "ring_radius",
    "vol_20",
    "lr_z20",
]


@dataclass
class ModelStrategyResult:
    positions: pd.Series
    predictions: pd.Series
    coefficients: dict
    train_score: float
    test_score: float
    test_start_idx: int
    target_horizon: int


def build_target(features: pd.DataFrame, horizon: int = 12) -> pd.Series:
    """Future log-return over `horizon` bars.

    target[t] = log(close[t+h]) - log(close[t])

    Last `horizon` rows are NaN by construction.
    """
    log_close = np.log(features["close"].to_numpy(dtype=float))
    target = np.full(len(features), np.nan)
    target[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    return pd.Series(target, index=features.index, name=f"target_h{horizon}")


def model_strategy(
    features: pd.DataFrame,
    horizon: int = 12,
    train_frac: float = 0.5,
    feature_cols: Optional[list[str]] = None,
    ridge_alpha: float = 1.0,
    signal_scale: float = 3.0,
    min_trade_size: float = 0.15,
    return_diagnostics: bool = False,
) -> pd.Series | ModelStrategyResult:
    """Train a Ridge regression on train slice → use predictions as positions.

    Args:
        features:        Output of build_feature_vector. Must contain 'close'
                         and the columns listed in feature_cols.
        horizon:         Target prediction horizon in bars.
        train_frac:      Fraction of the input used to fit the model.
                         The remaining (1 - train_frac) is the OOS test slice.
        feature_cols:    Subset of features to use. Defaults to the 9-feature set.
        ridge_alpha:     L2 regularisation strength.
        signal_scale:    Multiplier applied to standardised prediction before
                         clipping to [-1, 1]. Higher → more aggressive sizing.
        min_trade_size:  Dead-band threshold. |Δposition| < threshold → hold.
        return_diagnostics: If True, return ModelStrategyResult; else only positions.

    Returns:
        Position series in [-1, 1] indexed like features (zeros on train slice).
        If return_diagnostics, returns the full ModelStrategyResult.
    """
    cols = feature_cols or DEFAULT_FEATURES
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if "close" not in features.columns:
        raise KeyError("features must contain 'close' column for target construction")

    target = build_target(features, horizon=horizon)
    n = len(features)
    n_train = int(n * train_frac)
    # Drop the last `horizon` rows from the train slice — their targets are NaN
    train_end = n_train - horizon
    if train_end < 100:
        raise ValueError(
            f"Train slice too small ({train_end} rows). "
            f"Need at least 100 after dropping {horizon}-bar warmup."
        )

    X = features[cols].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)

    X_train = X[:train_end]
    y_train = y[:train_end]

    # Drop any NaN rows (from feature warmup) before fitting
    train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    if train_mask.sum() < 100:
        raise ValueError("Too few valid training rows after NaN removal.")

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train[train_mask], y_train[train_mask])

    train_score = float(model.score(X_train[train_mask], y_train[train_mask]))

    # Predict over the full input. Zero out training-slice positions.
    pred_all = np.full(n, np.nan)
    valid_mask = ~np.isnan(X).any(axis=1)
    pred_all[valid_mask] = model.predict(X[valid_mask])

    # Standardise predictions using train-slice stats only
    train_pred = pred_all[:train_end]
    train_pred_valid = train_pred[~np.isnan(train_pred)]
    mu = float(train_pred_valid.mean())
    sd = float(train_pred_valid.std(ddof=0)) or 1.0
    pred_std = (pred_all - mu) / sd

    # Test score (out-of-sample R²) — descriptive only, not used for decisions
    test_slice = slice(n_train, n - horizon)
    test_mask = ~(np.isnan(X[test_slice]).any(axis=1) | np.isnan(y[test_slice]))
    if test_mask.sum() > 50:
        test_score = float(
            model.score(X[test_slice][test_mask], y[test_slice][test_mask])
        )
    else:
        test_score = float("nan")

    # Position = clip(scale × standardised_prediction, -1, 1), zeroed on train
    raw_pos = signal_scale * pred_std
    desired = np.clip(raw_pos, -1.0, 1.0)
    desired = np.where(np.isnan(desired), 0.0, desired)
    desired[:n_train] = 0.0  # zero out train slice — no trading there

    # Dead-band
    held = np.zeros(n)
    current = 0.0
    for i in range(n):
        if abs(desired[i] - current) >= min_trade_size:
            current = desired[i]
        held[i] = current

    positions = pd.Series(held, index=features.index, name="position")

    if not return_diagnostics:
        return positions

    coefs = {c: float(w) for c, w in zip(cols, model.coef_)}
    return ModelStrategyResult(
        positions=positions,
        predictions=pd.Series(pred_std, index=features.index, name="prediction"),
        coefficients=coefs,
        train_score=train_score,
        test_score=test_score,
        test_start_idx=n_train,
        target_horizon=horizon,
    )
