"""Experiment 05: First direction prediction model using cycle phase features.

Hypothesis:
    A simple logistic regression on [sin/cos theta_24h, ring_radius, vol_20, lr_z20]
    achieves accuracy > 53% OOS on sign(log_return(t+12)).

Dataset:
    build_feature_vector(df, train_size=27000) — 19,486 OOS rows (Feb 2024 - Apr 2026).
    These rows were NEVER seen by the UMAP model.

Temporal split:
    Train: first 15,000 OOS rows (~Feb 2024 - ~Aug 2025)
    Test:  last ~4,474 rows (~Aug 2025 - Apr 2026)

Target: sign(log_return(t+12)) = direction 12h ahead.

Baseline: always predict +1 (majority class, ~50.6% accuracy).
Target:   accuracy > 53% OOS.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd

from eso.signals.feature_vector import build_feature_vector, FEATURE_COLUMNS
from eso.signals.direction_model import (
    DirectionModel,
    build_direction_dataset,
    CYCLE_FEATURES,
    DirectionResult,
)


OUT_DIR   = pathlib.Path("reports/direction_model_v1")
DATA_PATH = pathlib.Path("data/BTCUSDT_1h.csv")
TRAIN_SIZE_UMAP = 27_000   # UMAP training window (fixed, no lookahead)
TRAIN_SIZE_ML   = 15_000   # ML model training window (first N OOS rows)
HORIZON         = 12       # bars ahead to predict (= 12h)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw: {len(df):,} bars")

    print("Building causal feature vector (UMAP train_size=27000)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fv = build_feature_vector(df, train_size=TRAIN_SIZE_UMAP, smooth_windows=(6, 24, 72))
    print(f"  Feature vector: {fv.shape}")
    print(f"  Date range: {fv['timestamp'].iloc[0]} -> {fv['timestamp'].iloc[-1]}")

    print("Building direction dataset (horizon=12)...")
    X, y = build_direction_dataset(fv, horizon=HORIZON)
    print(f"  Total samples: {len(X):,}")
    print(f"  Class balance: +1={( y>0).mean():.3f}  -1={(y<0).mean():.3f}")

    # Temporal split
    X_train, y_train = X.iloc[:TRAIN_SIZE_ML], y.iloc[:TRAIN_SIZE_ML]
    X_test,  y_test  = X.iloc[TRAIN_SIZE_ML:],  y.iloc[TRAIN_SIZE_ML:]

    print(f"\n  Train: {len(X_train):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    if "timestamp" in fv.columns:
        train_ts = fv["timestamp"].loc[X_train.index]
        test_ts  = fv["timestamp"].loc[X_test.index]
        print(f"  Train period: {train_ts.iloc[0]} -> {train_ts.iloc[-1]}")
        print(f"  Test period:  {test_ts.iloc[0]} -> {test_ts.iloc[-1]}")

    return X_train, y_train, X_test, y_test


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_model(X_train, y_train, X_test, y_test, model_type: str) -> DirectionResult:
    print(f"\n=== {model_type.upper()} ===")
    poly_order = 2 if model_type == "logistic_poly" else 1
    actual_type = "logistic" if model_type == "logistic_poly" else model_type
    model = DirectionModel(model_type=actual_type, horizon=HORIZON, poly_order=poly_order)
    model.fit(X_train, y_train)

    result = model.evaluate(X_test, y_test)
    result.n_train = len(X_train)

    print(f"  Accuracy:      {result.accuracy:.4f}  ({result.accuracy*100:.2f}%)")
    print(f"  Baseline:      {result.baseline_accuracy:.4f}  ({result.baseline_accuracy*100:.2f}%)")
    print(f"  vs Baseline:   {result.accuracy_vs_baseline:+.4f}  ({result.accuracy_vs_baseline*100:+.2f}pp)")
    print(f"  Prec up/dn:    {result.precision_up:.3f} / {result.precision_down:.3f}")
    print(f"  Recall up/dn:  {result.recall_up:.3f} / {result.recall_down:.3f}")
    verdict = "PASS" if result.accuracy > 0.53 else ("MARGINAL" if result.accuracy > 0.51 else "FAIL")
    print(f"  Verdict:       {verdict}  (target > 53%)")

    if result.feature_importances:
        print(f"  Top feature importances (by |coef|):")
        sorted_imp = sorted(result.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_imp[:5]:
            print(f"    {name:20s} {val:+.4f}")

    return result


# ── Sub-period stability ───────────────────────────────────────────────────────

def run_stability_check(X_train, y_train, X_test, y_test) -> dict:
    """Check if accuracy is stable across first/second halves of test period."""
    model = DirectionModel(model_type="logistic", horizon=HORIZON)
    model.fit(X_train, y_train)

    n = len(X_test)
    h1_result = model.evaluate(X_test.iloc[:n//2], y_test.iloc[:n//2])
    h2_result = model.evaluate(X_test.iloc[n//2:], y_test.iloc[n//2:])

    print(f"\n--- Stability check (logistic) ---")
    print(f"  First half:  accuracy={h1_result.accuracy:.4f}  n={n//2}")
    print(f"  Second half: accuracy={h2_result.accuracy:.4f}  n={n-n//2}")
    print(f"  Std: {np.std([h1_result.accuracy, h2_result.accuracy]):.4f}")

    return {
        "first_half_accuracy": h1_result.accuracy,
        "second_half_accuracy": h2_result.accuracy,
        "stability": "stable" if abs(h1_result.accuracy - h2_result.accuracy) < 0.03 else "unstable",
    }


# ── Feature ablation ──────────────────────────────────────────────────────────

def run_ablation(X_train, y_train, X_test, y_test) -> dict:
    """Test accuracy with each feature group removed to assess contribution."""
    print(f"\n--- Feature ablation (logistic) ---")
    all_cols = list(X_train.columns)
    full_model = DirectionModel(model_type="logistic")
    full_model.fit(X_train, y_train)
    full_acc = full_model.evaluate(X_test, y_test).accuracy
    print(f"  All features: {full_acc:.4f}")

    groups = {
        "cycle_6h":  ["sin_theta_6h",  "cos_theta_6h"],
        "cycle_24h": ["sin_theta_24h", "cos_theta_24h"],
        "cycle_72h": ["sin_theta_72h", "cos_theta_72h"],
        "ring_radius": ["ring_radius"],
        "financial":   ["vol_20", "lr_z20"],
    }

    ablation = {}
    for grp_name, grp_cols in groups.items():
        remaining = [c for c in all_cols if c not in grp_cols]
        if not remaining:
            continue
        m = DirectionModel(model_type="logistic")
        m.fit(X_train[remaining], y_train)
        acc = m.evaluate(X_test[remaining], y_test).accuracy
        delta = acc - full_acc
        ablation[grp_name] = {"accuracy": acc, "delta_vs_full": round(delta, 4)}
        print(f"  w/o {grp_name:12s}: {acc:.4f}  (delta={delta:+.4f})")

    return ablation


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "direction_v1_result.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

def check_causal_correlation(X: pd.DataFrame, y_full: pd.Series, fv: pd.DataFrame) -> dict:
    """Verify true causal correlation between cycle features and future returns."""
    print("\n--- Causal correlation check (center=False, no lookahead) ---")
    import numpy as np

    close = fv["close"].to_numpy(dtype=float)
    n = len(close)
    future_lr = np.full(n, np.nan)
    future_lr[:n-12] = np.log(close[12:] / close[:n-12])
    fv2 = fv.copy()
    fv2["_future_lr"] = future_lr
    fv2 = fv2.dropna(subset=["_future_lr"])

    results = {}
    for col in ["sin_theta_24h", "cos_theta_24h", "sin_theta_6h", "sin_theta_72h", "ring_radius"]:
        if col not in fv2.columns:
            continue
        r = float(np.corrcoef(fv2[col].values, fv2["_future_lr"].values)[0, 1])
        results[col] = round(r, 4)
        print(f"  r({col}, future_12h) = {r:+.4f}")
    print(f"  Note: fit_and_evaluate uses center=True (lookahead ~12h) -> r=0.320")
    print(f"  build_feature_vector uses center=False (causal) -> see above")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 05: Direction Model v1")
    print("=" * 60)

    X_train, y_train, X_test, y_test = prepare_data()

    # First check actual causal correlation
    df = pd.read_csv(DATA_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fv = build_feature_vector(df, train_size=TRAIN_SIZE_UMAP, smooth_windows=(6, 24, 72))
    X_full, y_full = build_direction_dataset(fv, horizon=HORIZON)
    causal_corr = check_causal_correlation(X_full, y_full, fv)

    logistic_result = run_model(X_train, y_train, X_test, y_test, "logistic")
    logistic_poly_result = run_model(X_train, y_train, X_test, y_test, "logistic_poly")
    rf_result     = run_model(X_train, y_train, X_test, y_test, "rf")

    lgbm_result = None
    try:
        lgbm_result = run_model(X_train, y_train, X_test, y_test, "lgbm")
    except ImportError:
        print("\nLightGBM not available — skipping")

    stability = run_stability_check(X_train, y_train, X_test, y_test)
    ablation  = run_ablation(X_train, y_train, X_test, y_test)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Logistic accuracy:  {logistic_result.accuracy:.4f}  (baseline={logistic_result.baseline_accuracy:.4f})")
    if lgbm_result:
        print(f"  LightGBM accuracy:  {lgbm_result.accuracy:.4f}")
    print(f"  Stability:          {stability['stability']}")

    logistic_verdict = "PASS" if logistic_result.accuracy > 0.53 else (
        "MARGINAL" if logistic_result.accuracy > 0.51 else "FAIL")
    print(f"  Verdict: {logistic_verdict}  (target > 53%)")
    print("=" * 60)

    save_results({
        "experiment": "05_direction_model_v1",
        "data": str(DATA_PATH),
        "umap_train_size": TRAIN_SIZE_UMAP,
        "ml_train_size": TRAIN_SIZE_ML,
        "horizon_bars": HORIZON,
        "critical_finding": {
            "lookahead_bug": "fit_and_evaluate used center=True smoothing (win/2 lookahead). Fixed to center=False.",
            "corrected_r_oos": -0.031,
            "old_r_oos_was_artifact": "r=0.320 (center=True, win=24, h=12) was 100% lookahead artifact",
            "causal_correlations_sin_theta_24h": causal_corr.get("sin_theta_24h", None),
            "ring_radius_future_vol_12h": -0.2867,
            "ring_radius_interpretation": "Large radius = market on ring = low future volatility",
        },
        "logistic": {
            "n_train": logistic_result.n_train,
            "n_test": logistic_result.n_test,
            "accuracy": logistic_result.accuracy,
            "baseline_accuracy": logistic_result.baseline_accuracy,
            "accuracy_vs_baseline": logistic_result.accuracy_vs_baseline,
            "precision_up": logistic_result.precision_up,
            "precision_down": logistic_result.precision_down,
            "recall_up": logistic_result.recall_up,
            "recall_down": logistic_result.recall_down,
            "feature_importances": logistic_result.feature_importances,
            "verdict": logistic_verdict,
        },
        "logistic_poly": {
            "accuracy": logistic_poly_result.accuracy,
            "accuracy_vs_baseline": logistic_poly_result.accuracy_vs_baseline,
        },
        "rf": {
            "accuracy": rf_result.accuracy,
            "accuracy_vs_baseline": rf_result.accuracy_vs_baseline,
            "feature_importances": rf_result.feature_importances,
        },
        "lgbm": {
            "accuracy": lgbm_result.accuracy if lgbm_result else None,
            "accuracy_vs_baseline": lgbm_result.accuracy_vs_baseline if lgbm_result else None,
        } if lgbm_result else None,
        "stability": stability,
        "ablation": ablation,
        "causal_correlations": causal_corr,
    })
