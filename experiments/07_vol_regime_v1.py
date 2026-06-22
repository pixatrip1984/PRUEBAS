"""Experiment 07: Volatility regime classification using cycle phase features.

Target: predict whether realized_vol_12h > median(training_vol_12h).
        1 = high-vol regime, 0 = low-vol regime.

Hypothesis: accuracy > 58% OOS using [sin/cos_theta_24h, ring_radius, vol_20].

Baseline: majority class (50% by construction since threshold = training median).

Expected from regression r=+0.398:
  accuracy ~ (1 + |r| * sqrt(2/pi)) / 2 ~ 0.66 for perfectly Gaussian signals.
  Actual financial data is non-Gaussian so real accuracy may differ.

OOS: Feb 2024 - Apr 2026 (19,474 rows after target computation)
Train: first 15,000 rows | Test: last 4,474 rows
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd

from eso.signals.feature_vector import build_feature_vector
from eso.signals.volatility_model import (
    VolatilityModel,
    VolatilityRegimeClassifier,
    build_vol_dataset,
    VOL_FEATURES,
)


OUT_DIR         = pathlib.Path("reports/vol_regime_v1")
DATA_PATH       = pathlib.Path("data/BTCUSDT_1h.csv")
TRAIN_SIZE_UMAP = 27_000
TRAIN_SIZE_ML   = 15_000
HORIZON         = 12


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fv = build_feature_vector(df, train_size=TRAIN_SIZE_UMAP, smooth_windows=(6, 24, 72))

    feat_cols = [c for c in VOL_FEATURES if c in fv.columns]
    X_all, y_all = build_vol_dataset(fv, horizon=HORIZON, feature_cols=feat_cols)

    X_train, y_train = X_all.iloc[:TRAIN_SIZE_ML], y_all.iloc[:TRAIN_SIZE_ML]
    X_test,  y_test  = X_all.iloc[TRAIN_SIZE_ML:],  y_all.iloc[TRAIN_SIZE_ML:]

    threshold = float(y_train.median())
    print(f"Training threshold (median rv_12h): {threshold:.6f}")
    print(f"Test class balance: high={((y_test > threshold).mean()):.3f}  low={(~(y_test > threshold)).mean():.3f}")
    return X_train, y_train, X_test, y_test, threshold


def run_classifier(X_train, y_train, X_test, y_test, model_type, feature_cols, label):
    clf = VolatilityRegimeClassifier(model_type=model_type, horizon=HORIZON)
    clf.fit(X_train[feature_cols], y_train)
    result = clf.evaluate(X_test[feature_cols], y_test)
    result.n_train = len(X_train)

    verdict = "PASS" if result.accuracy > 0.58 else ("MARGINAL" if result.accuracy > 0.55 else "FAIL")
    print(f"\n{label}")
    print(f"  Accuracy:      {result.accuracy:.4f}  ({result.accuracy*100:.1f}%)")
    print(f"  Baseline:      {result.baseline_accuracy:.4f}  ({result.baseline_accuracy*100:.1f}%)")
    print(f"  vs Baseline:   {result.accuracy_vs_baseline:+.4f}  ({result.accuracy_vs_baseline*100:+.1f}pp)")
    print(f"  Prec high/low: {result.precision_high:.3f} / {result.precision_low:.3f}")
    print(f"  Recall high/low: {result.recall_high:.3f} / {result.recall_low:.3f}")
    print(f"  Verdict: {verdict}  (target > 58%)")
    return result


def stability_check(X_train, y_train, X_test, y_test, feature_cols):
    clf = VolatilityRegimeClassifier(model_type="logistic")
    clf.fit(X_train[feature_cols], y_train)
    n = len(X_test)
    r1 = clf.evaluate(X_test[feature_cols].iloc[:n//2], y_test.iloc[:n//2])
    r2 = clf.evaluate(X_test[feature_cols].iloc[n//2:], y_test.iloc[n//2:])
    print(f"\n--- Stability (logistic, phase_24h+ring+vol20) ---")
    print(f"  First half:  acc={r1.accuracy:.4f}  n={n//2}")
    print(f"  Second half: acc={r2.accuracy:.4f}  n={n-n//2}")
    stable = abs(r1.accuracy - r2.accuracy) < 0.03
    print(f"  Stability: {'stable' if stable else 'unstable (delta > 3pp)'}")
    return {"first_half": r1.accuracy, "second_half": r2.accuracy, "stable": stable}


def regression_to_binary_accuracy(X_train, y_train, X_test, y_test, feature_cols, threshold):
    """Use regression model predictions thresholded at training median."""
    reg = VolatilityModel(model_type="ridge")
    reg.fit(X_train[feature_cols], y_train)
    y_pred_cont = reg.predict(X_test[feature_cols])
    y_pred_bin = (y_pred_cont > threshold).astype(int)
    y_true_bin = (y_test.values > threshold).astype(int)
    acc = float((y_pred_bin == y_true_bin).mean())
    print(f"\n  Regression->Binary (threshold at train median): accuracy={acc:.4f}")
    return acc


if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 07: Volatility Regime Classifier v1")
    print("=" * 65)

    X_train, y_train, X_test, y_test, threshold = prepare_data()

    # Majority baseline
    print(f"\n=== BASELINE: majority class ===")
    majority_acc = 0.5  # threshold = median → balanced by construction
    print(f"  Expected baseline: {majority_acc:.3f} (balanced by design)")

    feat_cols = [c for c in VOL_FEATURES if c in X_train.columns]
    vol_only    = ["vol_20"]
    phase_vol   = ["sin_theta_24h", "cos_theta_24h", "vol_20"]
    phase_ring  = ["sin_theta_24h", "cos_theta_24h", "ring_radius", "vol_20"]
    phase_all   = ["sin_theta_24h", "cos_theta_24h", "sin_theta_72h", "cos_theta_72h",
                   "ring_radius", "vol_20", "lr_z20"]

    configs = [
        ("logistic", vol_only,   "LOGISTIC: vol_20 only"),
        ("logistic", phase_vol,  "LOGISTIC: phase_24h + vol_20"),
        ("logistic", phase_ring, "LOGISTIC: phase_24h + ring_radius + vol_20"),
        ("logistic", phase_all,  "LOGISTIC: all phase + vol_20"),
        ("rf",       phase_ring, "RF:       phase_24h + ring_radius + vol_20"),
        ("rf",       phase_all,  "RF:       all phase + vol_20"),
    ]

    results = {}
    for mtype, cols, label in configs:
        r = run_classifier(X_train, y_train, X_test, y_test, mtype, cols, label)
        results[label] = {
            "model_type": mtype, "feature_cols": cols,
            "accuracy": r.accuracy, "baseline": r.baseline_accuracy,
            "vs_baseline": r.accuracy_vs_baseline,
            "precision_high": r.precision_high, "recall_high": r.recall_high,
            "precision_low": r.precision_low, "recall_low": r.recall_low,
            "pass": r.accuracy > 0.58,
        }

    # Regression->Binary comparison
    reg_bin_acc = regression_to_binary_accuracy(
        X_train, y_train, X_test, y_test, phase_ring, threshold)

    stability = stability_check(X_train, y_train, X_test, y_test, phase_ring)

    best_label = max(results, key=lambda k: results[k]["accuracy"])
    best_acc = results[best_label]["accuracy"]

    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  Best model:  {best_label}")
    print(f"  Best acc:    {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"  Baseline:    0.500  (50.0%)")
    print(f"  Reg->Binary: {reg_bin_acc:.4f}")
    any_pass = any(r["pass"] for r in results.values())
    print(f"  H1 (any model > 58%): {'PASS' if any_pass else 'FAIL'}")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "regime_v1_result.json").write_text(json.dumps({
        "experiment": "07_vol_regime_v1",
        "data": str(DATA_PATH), "umap_train_size": TRAIN_SIZE_UMAP,
        "ml_train_size": TRAIN_SIZE_ML, "horizon_bars": HORIZON,
        "threshold_median": threshold,
        "models": results,
        "regression_to_binary_accuracy": reg_bin_acc,
        "stability": stability,
        "conclusion": {
            "best_model": best_label, "best_accuracy": best_acc,
            "beats_58pct_target": any_pass,
        },
    }, indent=2, default=str))
    print(f"\nResults saved to {OUT_DIR / 'regime_v1_result.json'}")
