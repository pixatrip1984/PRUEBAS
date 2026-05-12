"""Direction prediction model using causal cycle features.

Predicts the sign of log_return(t + horizon) using the S1 cycle phase
features produced by build_feature_vector().

This is the first downstream ML model in the ESO pipeline. It uses only
OOS features (never saw the training window used by UMAP) and is evaluated
with a strict temporal holdout split.

Usage:
    from eso.signals.direction_model import DirectionModel, build_direction_dataset
    from eso.signals.feature_vector import build_feature_vector

    fv = build_feature_vector(df, train_size=27000)
    X, y = build_direction_dataset(fv, horizon=12)

    model = DirectionModel(model_type="logistic")
    model.fit(X.iloc[:15000], y.iloc[:15000])
    result = model.evaluate(X.iloc[15000:], y.iloc[15000:])
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


CYCLE_FEATURES = [
    "sin_theta_6h",  "cos_theta_6h",
    "sin_theta_24h", "cos_theta_24h",
    "sin_theta_72h", "cos_theta_72h",
    "ring_radius",
    "vol_20",
    "lr_z20",
]


@dataclass
class DirectionResult:
    """Evaluation results for the direction model."""
    model_type: str
    n_train: int
    n_test: int
    horizon: int
    # Accuracy
    accuracy: float
    baseline_accuracy: float      # majority-class baseline
    accuracy_vs_baseline: float   # accuracy - baseline
    # Per-class
    precision_up: float
    precision_down: float
    recall_up: float
    recall_down: float
    # Signal quality
    feature_importances: dict     # feature -> weight (logistic) or importance


def build_direction_dataset(
    fv: pd.DataFrame,
    horizon: int = 12,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from a feature vector DataFrame.

    Target: sign of cumulative log_return over next `horizon` bars.
    Rows where target is 0 (flat) are dropped (very rare).

    Args:
        fv:           Output of build_feature_vector() with 'close' column.
        horizon:      Look-ahead in bars.
        feature_cols: Feature columns to use (default: CYCLE_FEATURES).

    Returns:
        (X, y) where y in {-1, +1}.
    """
    if feature_cols is None:
        feature_cols = [c for c in CYCLE_FEATURES if c in fv.columns]

    close = fv["close"].to_numpy(dtype=float)
    n = len(close)

    # Forward log return: log(close[t+h] / close[t])
    future_lr = np.full(n, np.nan)
    future_lr[: n - horizon] = np.log(close[horizon:] / close[:n-horizon])

    fv2 = fv.copy()
    fv2["_target_lr"] = future_lr
    fv2["_target"] = np.sign(fv2["_target_lr"])

    # Drop last horizon rows (no target) and flat returns
    fv2 = fv2.dropna(subset=["_target_lr"])
    fv2 = fv2[fv2["_target"] != 0]

    X = fv2[feature_cols]
    y = fv2["_target"].astype(int)
    return X, y


class DirectionModel:
    """Directional return predictor using cycle phase features.

    Wraps sklearn LogisticRegression (default) or LightGBM.
    Scales features with RobustScaler fitted on training data only.

    Args:
        model_type: "logistic" (default), "rf" (Random Forest), or "lgbm".
        horizon:    Look-ahead in bars (informational only — target must
                    be pre-computed via build_direction_dataset).
        C:          Regularisation for logistic regression.
        seed:       Reproducibility.
        poly_order: If > 1, adds cross-terms up to this degree for cycle features
                    (helps capture the periodic sin/cos relationship).
    """

    def __init__(
        self,
        model_type: str = "logistic",
        horizon: int = 12,
        C: float = 0.1,
        seed: int = 42,
        poly_order: int = 1,
    ) -> None:
        self.model_type = model_type
        self.horizon = horizon
        self.C = C
        self.seed = seed
        self.poly_order = poly_order
        self._scaler = RobustScaler()
        self._model = None
        self._feature_cols: list[str] = []

    def _add_poly_features(self, X_np: np.ndarray) -> np.ndarray:
        """Add degree-2 cross-terms for sin/cos cycle features."""
        if self.poly_order < 2:
            return X_np
        cycle_idx = [i for i, c in enumerate(self._feature_cols)
                     if c.startswith("sin_") or c.startswith("cos_")]
        extras = []
        for i in cycle_idx:
            for j in cycle_idx:
                if i <= j:
                    extras.append(X_np[:, i] * X_np[:, j])
        if extras:
            return np.column_stack([X_np] + extras)
        return X_np

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionModel":
        """Fit scaler + classifier on training data."""
        self._feature_cols = list(X.columns)
        X_raw = self._scaler.fit_transform(X.to_numpy(dtype=float))
        X_scaled = self._add_poly_features(X_raw)

        if self.model_type == "lgbm":
            try:
                import lightgbm as lgb
                self._model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=50,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.seed,
                    verbose=-1,
                )
                self._model.fit(X_scaled, y)
            except ImportError:
                raise ImportError("lightgbm required: pip install lightgbm")
        elif self.model_type == "rf":
            self._model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=100,
                random_state=self.seed,
                n_jobs=-1,
            )
            self._model.fit(X_scaled, y)
        else:
            self._model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.seed,
                solver="lbfgs",
            )
            self._model.fit(X_scaled, y)
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X_raw = self._scaler.transform(X[self._feature_cols].to_numpy(dtype=float))
        return self._add_poly_features(X_raw)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted labels (-1 or +1)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        return self._model.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of +1 direction."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        proba = self._model.predict_proba(self._transform(X))
        # classes_ is sorted, find index of +1
        classes = list(self._model.classes_)
        idx_up = classes.index(1)
        return proba[:, idx_up]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> DirectionResult:
        """Evaluate on test data and return structured results."""
        y_pred = self.predict(X)
        y_true = y.to_numpy(dtype=int)

        accuracy = float((y_pred == y_true).mean())
        majority = int(np.bincount(y_true + 1).argmax()) - 1  # shift back
        baseline_accuracy = float((y_true == majority).mean())

        # Per-class metrics
        # Precision: fraction of predicted class that is correct
        # Recall:    fraction of actual class that is correctly predicted
        pred_up  = y_pred == 1
        pred_dn  = y_pred == -1
        true_up  = y_true == 1
        true_dn  = y_true == -1
        prec_up  = float((y_true[pred_up] == 1).mean())   if pred_up.any()  else 0.0
        prec_dn  = float((y_true[pred_dn] == -1).mean())  if pred_dn.any()  else 0.0
        rec_up   = float((y_pred[true_up] == 1).mean())   if true_up.any()  else 0.0
        rec_dn   = float((y_pred[true_dn] == -1).mean())  if true_dn.any()  else 0.0

        # Feature importances
        importances = {}
        if self.model_type == "logistic" and self._model is not None:
            coefs = self._model.coef_[0]
            for name, coef in zip(self._feature_cols, coefs):
                importances[name] = round(float(coef), 4)
        elif self.model_type in ("lgbm", "rf") and self._model is not None:
            for name, imp in zip(self._feature_cols, self._model.feature_importances_[:len(self._feature_cols)]):
                importances[name] = round(float(imp), 4)

        return DirectionResult(
            model_type=self.model_type,
            n_train=0,
            n_test=len(y_true),
            horizon=self.horizon,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            accuracy_vs_baseline=accuracy - baseline_accuracy,
            precision_up=prec_up,
            precision_down=prec_dn,
            recall_up=rec_up,
            recall_down=rec_dn,
            feature_importances=importances,
        )
