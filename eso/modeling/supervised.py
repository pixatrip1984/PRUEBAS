"""Reproducible supervised evaluation runs for model-building agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eso.contracts import assert_valid_contract
from eso.data.features import build_financial_features, feature_column_groups
from eso.signals.direction_model import CYCLE_FEATURES, DirectionModel, build_direction_dataset
from eso.signals.volatility_model import (
    VOL_FEATURES,
    VolatilityModel,
    VolatilityRegimeClassifier,
    build_vol_dataset,
)


DEFAULT_TRAIN_FRACTION = 0.7
TASKS = {"volatility", "regime", "direction"}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _frame_fingerprint(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    payload = json.dumps(cols, separators=(",", ":")).encode("utf-8") + b"\0"
    payload += np.ascontiguousarray(numeric).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _resolve_train_size(n: int, train_size: int | None, train_fraction: float) -> int:
    if n < 3:
        raise ValueError("not enough samples for a train/test split")
    if train_size is None:
        train_size = int(n * float(train_fraction))
    train_size = int(train_size)
    if train_size <= 0 or train_size >= n:
        raise ValueError(f"train_size must be in [1, {n - 1}], got {train_size}")
    return train_size


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    """Split X/y without shuffling and return train/test slices."""
    n = min(len(X), len(y))
    if n != len(X) or n != len(y):
        X = X.iloc[:n]
        y = y.iloc[:n]
    split = _resolve_train_size(n, train_size=train_size, train_fraction=train_fraction)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], split


def build_feature_frame(
    raw: pd.DataFrame,
    feature_mode: str = "raw",
    columns: list[str] | None = None,
    close_col: str = "close",
    umap_train_size: int = 27000,
    smooth_windows: tuple[int, ...] = (6, 24, 72),
) -> pd.DataFrame:
    """Return a model-ready feature frame with a close column for target building.

    `feature_mode=raw` selects numeric input columns directly. Financial modes
    use `build_financial_features()` and preserve original row alignment for
    close/timestamp metadata.
    """
    if close_col not in raw.columns:
        raise ValueError(f"close column not found: {close_col}")

    if feature_mode == "cycle":
        from eso.signals.feature_vector import build_feature_vector

        out = build_feature_vector(
            raw.rename(columns={close_col: "close"}) if close_col != "close" else raw,
            train_size=umap_train_size,
            smooth_windows=smooth_windows,
        )
        if columns:
            keep = [c for c in columns if c in out.columns]
            missing = [c for c in columns if c not in out.columns]
            if missing:
                raise ValueError(f"cycle feature columns not found: {missing}")
            for required in ["close", "timestamp"]:
                if required in out.columns and required not in keep:
                    keep.append(required)
            out = out[keep]
        return out

    if feature_mode == "raw":
        selected = columns or raw.select_dtypes(include=["number"]).columns.tolist()
        missing = [c for c in selected if c not in raw.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        out = raw[selected].copy()
        if close_col not in out.columns:
            out[close_col] = raw[close_col].values
        if "timestamp" in raw.columns:
            out.insert(0, "timestamp", raw["timestamp"].values)
        return out

    feat = build_financial_features(raw)
    groups = feature_column_groups()
    if feature_mode in groups:
        selected = columns or [c for c in groups[feature_mode] if c in feat.columns]
    elif feature_mode == "full":
        selected = columns or feat.columns.tolist()
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    missing = [c for c in selected if c not in feat.columns]
    if missing:
        raise ValueError(f"feature columns not found after feature engineering: {missing}")

    out = feat[selected].copy()
    out[close_col] = pd.to_numeric(raw.loc[out.index, close_col], errors="coerce").values
    if "timestamp" in raw.columns:
        out.insert(0, "timestamp", raw.loc[out.index, "timestamp"].values)
    return out


def _default_features(fv: pd.DataFrame, task: str) -> list[str]:
    if task in {"volatility", "regime"}:
        preferred = VOL_FEATURES + ["log_return", "volume_imbalance", "vwap_dev"]
    else:
        preferred = CYCLE_FEATURES + ["log_return", "volume_imbalance", "vwap_dev"]
    cols = [c for c in preferred if c in fv.columns]
    if cols:
        return cols
    return [
        c for c in fv.select_dtypes(include=["number"]).columns
        if c not in {"close"} and not c.startswith("future_")
    ]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _evaluate_volatility(
    fv: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    train_size: int | None,
    train_fraction: float,
    model_type: str,
) -> dict:
    X, y = build_vol_dataset(fv, horizon=horizon, feature_cols=feature_cols)
    X_train, y_train, X_test, y_test, split = temporal_split(X, y, train_size, train_fraction)

    model = VolatilityModel(model_type=model_type, horizon=horizon)
    model.fit(X_train, y_train)
    result = model.evaluate(X_test, y_test)
    result.n_train = len(X_train)

    y_true = y_test.to_numpy(dtype=float)
    mean_pred = np.full_like(y_true, float(y_train.mean()), dtype=float)
    baselines = {
        "train_mean_mae": float(np.mean(np.abs(mean_pred - y_true))),
    }
    if "vol_20" in X_test.columns:
        vol20 = X_test["vol_20"].to_numpy(dtype=float)
        baselines["vol_20_persistence_mae"] = float(np.mean(np.abs(vol20 - y_true)))

    return {
        "target": {"name": f"future_rv_{horizon}", "type": "regression", "horizon": horizon},
        "split": {"train_size": split, "test_size": len(X_test), "shuffle": False},
        "metrics": {
            "mae": result.mae,
            "rmse": result.rmse,
            "pearson_r": result.pearson_r,
            "mae_vs_train_mean": result.mae - baselines["train_mean_mae"],
            "mae_vs_vol_20_persistence": (
                result.mae - baselines["vol_20_persistence_mae"]
                if "vol_20_persistence_mae" in baselines else None
            ),
        },
        "baselines": baselines,
        "feature_importances": result.feature_importances,
    }


def _evaluate_regime(
    fv: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    train_size: int | None,
    train_fraction: float,
    model_type: str,
) -> dict:
    X, y_cont = build_vol_dataset(fv, horizon=horizon, feature_cols=feature_cols)
    X_train, y_train, X_test, y_test, split = temporal_split(X, y_cont, train_size, train_fraction)

    classifier_type = "rf" if model_type == "rf" else "logistic"
    clf = VolatilityRegimeClassifier(model_type=classifier_type, horizon=horizon)
    clf.fit(X_train, y_train)
    result = clf.evaluate(X_test, y_test)
    result.n_train = len(X_train)
    return {
        "target": {
            "name": f"future_rv_{horizon}_above_train_median",
            "type": "classification",
            "horizon": horizon,
            "threshold": result.threshold,
        },
        "split": {"train_size": split, "test_size": len(X_test), "shuffle": False},
        "metrics": {
            "accuracy": result.accuracy,
            "baseline_accuracy": result.baseline_accuracy,
            "accuracy_vs_baseline": result.accuracy_vs_baseline,
            "precision_high": result.precision_high,
            "recall_high": result.recall_high,
            "precision_low": result.precision_low,
            "recall_low": result.recall_low,
        },
        "baselines": {"majority_class_accuracy": result.baseline_accuracy},
        "feature_importances": result.feature_importances,
    }


def _evaluate_direction(
    fv: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    train_size: int | None,
    train_fraction: float,
    model_type: str,
) -> dict:
    X, y = build_direction_dataset(fv, horizon=horizon, feature_cols=feature_cols)
    X_train, y_train, X_test, y_test, split = temporal_split(X, y, train_size, train_fraction)

    classifier_type = "rf" if model_type == "rf" else "logistic"
    model = DirectionModel(model_type=classifier_type, horizon=horizon)
    model.fit(X_train, y_train)
    result = model.evaluate(X_test, y_test)
    result.n_train = len(X_train)
    return {
        "target": {"name": f"future_direction_{horizon}", "type": "classification", "horizon": horizon},
        "split": {"train_size": split, "test_size": len(X_test), "shuffle": False},
        "metrics": {
            "accuracy": result.accuracy,
            "baseline_accuracy": result.baseline_accuracy,
            "accuracy_vs_baseline": result.accuracy_vs_baseline,
            "precision_up": result.precision_up,
            "recall_up": result.recall_up,
            "precision_down": result.precision_down,
            "recall_down": result.recall_down,
        },
        "baselines": {"majority_class_accuracy": result.baseline_accuracy},
        "feature_importances": result.feature_importances,
    }


def evaluate_supervised_task(
    fv: pd.DataFrame,
    task: str,
    horizon: int = 12,
    feature_cols: list[str] | None = None,
    train_size: int | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    model_type: str = "ridge",
    dataset_id: str = "anonymous",
) -> dict:
    """Evaluate one supervised task and return a portable manifest."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}. Choose one of {sorted(TASKS)}")
    if model_type == "auto":
        model_type = "ridge" if task == "volatility" else "logistic"
    elif task == "volatility" and model_type not in {"ridge", "rf"}:
        model_type = "ridge"
    elif task in {"regime", "direction"} and model_type not in {"logistic", "rf"}:
        model_type = "logistic"

    feature_cols = feature_cols or _default_features(fv, task)
    if not feature_cols:
        raise ValueError("no usable numeric feature columns found")

    if task == "volatility":
        result = _evaluate_volatility(fv, horizon, feature_cols, train_size, train_fraction, model_type)
    elif task == "regime":
        result = _evaluate_regime(fv, horizon, feature_cols, train_size, train_fraction, model_type)
    else:
        result = _evaluate_direction(fv, horizon, feature_cols, train_size, train_fraction, model_type)

    metrics = result["metrics"]
    primary = (
        "mae_vs_vol_20_persistence" if task == "volatility" and metrics.get("mae_vs_vol_20_persistence") is not None
        else "accuracy_vs_baseline" if task in {"regime", "direction"}
        else "mae_vs_train_mean"
    )
    return {
        "schema_version": "eso.model_run.v1",
        "dataset_id": dataset_id,
        "task": task,
        "model_type": model_type,
        "features": feature_cols,
        "feature_frame": {
            "rows": int(len(fv)),
            "columns": list(fv.columns),
            "sha256": _frame_fingerprint(fv.select_dtypes(include=["number"])),
        },
        "target": result["target"],
        "split": result["split"],
        "baselines": result["baselines"],
        "metrics": metrics,
        "primary_metric": primary,
        "feature_importances": result["feature_importances"],
        "causal_guardrails": [
            "Temporal split only; shuffle=False.",
            "Target is indexed at decision time t and may use future prices only as label.",
            "Feature builders must use only information available at t.",
        ],
    }


def write_model_run(manifest: dict, output_path: str | Path) -> str:
    assert_valid_contract(manifest, "eso.model_run.v1")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
