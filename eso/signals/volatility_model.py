"""Volatility prediction model using ring geometry + cycle phase features.

Key finding (2026-05-12): cycle phase features (sin/cos theta) and ring_radius
predict future VOLATILITY causally, even though they don't predict direction.

Causal correlations with realized_vol_12h (std of next 12 log_returns):
  vol_20:        r = +0.465  (GARCH-type persistence — strongest baseline)
  ring_radius:   r = -0.275  (large ring = structured market = low future vol)
  sin_theta_24h: r = -0.227  (phase feature has genuine volatility prediction)
  cos_theta_24h: r = +0.190

Partial r of ring_radius vs rv_12h (controlling for vol_20) = -0.065.
Partial r of sin_theta_24h vs rv_12h (controlling for vol_20) ≈ similar.

These signals can be combined into a regression model that beats vol_20-only.

Usage:
    from eso.signals.volatility_model import VolatilityModel, build_vol_dataset
    from eso.signals.feature_vector import build_feature_vector

    fv = build_feature_vector(df, train_size=27000)
    X, y = build_vol_dataset(fv, horizon=12)

    model = VolatilityModel(model_type="ridge")
    model.fit(X.iloc[:15000], y.iloc[:15000])
    result = model.evaluate(X.iloc[15000:], y.iloc[15000:])
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler


VOL_FEATURES = [
    "ring_radius",
    "vol_20",
    "sin_theta_24h", "cos_theta_24h",
    "sin_theta_6h",  "cos_theta_6h",
    "sin_theta_72h", "cos_theta_72h",
    "lr_z20",
]


@dataclass
class VolatilityResult:
    """Evaluation results for the volatility model."""
    model_type: str
    feature_set: str
    n_train: int
    n_test: int
    horizon: int
    # Metrics
    mae: float
    rmse: float
    pearson_r: float
    # vs baselines
    mae_vs_mean_baseline: float     # negative = better than mean
    mae_vs_garch0_baseline: float   # negative = better than vol_20 persistence
    # Partial correlations
    partial_r_vs_vol20: float       # ring_radius partial r (controlling for vol_20)
    # Feature importances
    feature_importances: dict


def build_vol_dataset(
    fv: pd.DataFrame,
    horizon: int = 12,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for volatility prediction.

    Target y: realized volatility over next `horizon` bars =
              std(log_return[t+1 : t+horizon+1], ddof=1).

    For horizon=1, |log_return(t+1)| is used instead of std.

    Args:
        fv:           Output of build_feature_vector() with 'close' column.
        horizon:      Prediction horizon in bars.
        feature_cols: Feature columns to use (default: VOL_FEATURES filtered to fv).

    Returns:
        (X, y) — aligned DataFrame and Series, no NaN rows.
    """
    if feature_cols is None:
        feature_cols = [c for c in VOL_FEATURES if c in fv.columns]

    close = fv["close"].to_numpy(dtype=float)
    n = len(close)
    lr = np.log(close[1:] / close[:-1])   # n-1 one-bar log returns

    # Realized vol over [t+1 : t+horizon+1]
    rv = np.full(n, np.nan)
    for i in range(n - horizon):
        if horizon == 1:
            rv[i] = abs(lr[i])
        else:
            rv[i] = float(np.std(lr[i:i+horizon], ddof=1))

    fv2 = fv.copy()
    fv2["_rv"] = rv
    fv2 = fv2.dropna(subset=["_rv"])

    X = fv2[feature_cols]
    y = fv2["_rv"]
    return X, y


class VolatilityModel:
    """Volatility predictor using ring geometry + cycle phase features.

    Supports ridge regression and random forest. All features are scaled
    with RobustScaler fitted on training data only.

    Args:
        model_type:   "ridge" (default), "rf" (Random Forest).
        horizon:      Look-ahead in bars (informational).
        alpha:        Ridge regularization.
        seed:         Reproducibility.
    """

    def __init__(
        self,
        model_type: str = "ridge",
        horizon: int = 12,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.model_type = model_type
        self.horizon = horizon
        self.alpha = alpha
        self.seed = seed
        self._scaler = RobustScaler()
        self._model = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VolatilityModel":
        self._feature_cols = list(X.columns)
        X_s = self._scaler.fit_transform(X.to_numpy(dtype=float))
        if self.model_type == "rf":
            self._model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=50,
                random_state=self.seed,
                n_jobs=-1,
            )
        else:
            self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_s, y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        X_s = self._scaler.transform(X[self._feature_cols].to_numpy(dtype=float))
        return self._model.predict(X_s)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        vol20_col: str = "vol_20",
    ) -> VolatilityResult:
        """Evaluate model vs mean-baseline and vol_20-persistence baseline."""
        y_true = y.to_numpy(dtype=float)
        y_pred = self.predict(X)

        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r    = float(np.corrcoef(y_pred, y_true)[0, 1])

        # Mean baseline
        mae_mean = float(np.mean(np.abs(y_true.mean() - y_true)))

        # GARCH-0 baseline: vol_20(t) as prediction of realized_vol(t+12)
        if vol20_col in X.columns:
            vol20 = X[vol20_col].to_numpy(dtype=float)
            mae_garch0 = float(np.mean(np.abs(vol20 - y_true)))
        else:
            mae_garch0 = float("nan")

        # Partial correlation: ring_radius vs y controlling for vol_20
        partial_r = float("nan")
        if "ring_radius" in X.columns and vol20_col in X.columns:
            rr = X["ring_radius"].to_numpy(dtype=float)
            v  = X[vol20_col].to_numpy(dtype=float)
            r_rf = float(np.corrcoef(rr, y_true)[0, 1])
            r_rv = float(np.corrcoef(rr, v)[0, 1])
            r_vf = float(np.corrcoef(v,  y_true)[0, 1])
            denom = np.sqrt(max(1e-12, (1 - r_rv**2) * (1 - r_vf**2)))
            partial_r = (r_rf - r_rv * r_vf) / denom

        # Feature importances
        importances: dict = {}
        if self.model_type == "ridge" and hasattr(self._model, "coef_"):
            for name, coef in zip(self._feature_cols, self._model.coef_):
                importances[name] = round(float(coef), 4)
        elif self.model_type == "rf" and hasattr(self._model, "feature_importances_"):
            for name, imp in zip(self._feature_cols, self._model.feature_importances_):
                importances[name] = round(float(imp), 4)

        return VolatilityResult(
            model_type=self.model_type,
            feature_set="+".join(self._feature_cols),
            n_train=0,
            n_test=len(y_true),
            horizon=self.horizon,
            mae=mae,
            rmse=rmse,
            pearson_r=r,
            mae_vs_mean_baseline=mae - mae_mean,
            mae_vs_garch0_baseline=mae - mae_garch0,
            partial_r_vs_vol20=partial_r,
            feature_importances=importances,
        )
