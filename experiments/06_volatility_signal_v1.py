"""Experiment 06: Volatility prediction using ring geometry + cycle phase.

Key hypothesis:
    Cycle phase features (sin/cos theta) and ring_radius predict realized
    volatility causally (all features center=False, no lookahead).

Causal correlations (pre-measured):
    vol_20:        r = +0.465  (GARCH persistence baseline)
    ring_radius:   r = -0.275  (large ring = structured market = low vol)
    sin_theta_24h: r = -0.227  (phase position predicts volatility)
    cos_theta_24h: r = +0.190

Target: realized_vol_12h = std(log_return[t+1:t+13])
OOS data: 19,486 rows (Feb 2024 - Apr 2026)
Train: first 15,000 OOS rows
Test:  last ~4,474 OOS rows

Hypotheses to test:
  H1: vol_20-only MAE < mean-baseline
  H2: ring_radius + vol_20 MAE < vol_20-only (partial info beyond GARCH)
  H3: all_features MAE < vol_20-only
  H4: RF beats Ridge (non-linear interactions between phase and vol)
  H5: improvement is stable across first/second half of test period
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
    VolatilityResult,
    build_vol_dataset,
    VOL_FEATURES,
)


OUT_DIR         = pathlib.Path("reports/volatility_signal_v1")
DATA_PATH       = pathlib.Path("data/BTCUSDT_1h.csv")
TRAIN_SIZE_UMAP = 27_000
TRAIN_SIZE_ML   = 15_000
HORIZON         = 12


# ── Data prep ─────────────────────────────────────────────────────────────────

def prepare_data():
    print("Loading and building feature vector...")
    df = pd.read_csv(DATA_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fv = build_feature_vector(df, train_size=TRAIN_SIZE_UMAP, smooth_windows=(6, 24, 72))
    print(f"  fv shape: {fv.shape}")

    X_all, y_all = build_vol_dataset(fv, horizon=HORIZON)
    print(f"  Dataset: {len(X_all):,} rows  target_mean={y_all.mean():.5f}  target_std={y_all.std():.5f}")
    print(f"  Date range: {fv['timestamp'].loc[X_all.index[0]]} -> {fv['timestamp'].loc[X_all.index[-1]]}")

    X_train, y_train = X_all.iloc[:TRAIN_SIZE_ML], y_all.iloc[:TRAIN_SIZE_ML]
    X_test,  y_test  = X_all.iloc[TRAIN_SIZE_ML:],  y_all.iloc[TRAIN_SIZE_ML:]

    train_ts = fv["timestamp"].loc[X_train.index]
    test_ts  = fv["timestamp"].loc[X_test.index]
    print(f"  Train: {len(X_train):,} rows  {train_ts.iloc[0]} -> {train_ts.iloc[-1]}")
    print(f"  Test:  {len(X_test):,} rows   {test_ts.iloc[0]} -> {test_ts.iloc[-1]}")
    return X_train, y_train, X_test, y_test


# ── Run one model config ──────────────────────────────────────────────────────

def run_model(
    X_train, y_train, X_test, y_test,
    model_type: str, feature_cols: list[str], label: str
) -> VolatilityResult:
    model = VolatilityModel(model_type=model_type, horizon=HORIZON)
    model.fit(X_train[feature_cols], y_train)
    result = model.evaluate(X_test[feature_cols], y_test)
    result.n_train = len(X_train)

    verdict_garch = "PASS" if result.mae_vs_garch0_baseline < 0 else "FAIL"
    print(f"\n{label}")
    print(f"  MAE:      {result.mae:.6f}")
    print(f"  RMSE:     {result.rmse:.6f}")
    print(f"  Pearson r:{result.pearson_r:+.3f}")
    print(f"  vs mean:  {result.mae_vs_mean_baseline:+.6f}")
    print(f"  vs GARCH0:{result.mae_vs_garch0_baseline:+.6f}  ({verdict_garch})")
    if not np.isnan(result.partial_r_vs_vol20):
        print(f"  partial_r:{result.partial_r_vs_vol20:+.3f}  (ring_r vs rv controlling for vol_20)")
    return result


# ── Phase correlation sweep ────────────────────────────────────────────────────

def phase_vol_correlation_sweep(X_test, y_test) -> dict:
    """Compute partial correlations of each cycle feature vs future vol, controlling for vol_20."""
    print("\n--- Partial correlations (controlling for vol_20) ---")
    v = X_test["vol_20"].to_numpy(dtype=float)
    y = y_test.to_numpy(dtype=float)
    r_vf = float(np.corrcoef(v, y)[0, 1])

    results = {}
    for col in X_test.columns:
        if col == "vol_20":
            continue
        f = X_test[col].to_numpy(dtype=float)
        r_ff = float(np.corrcoef(f, y)[0, 1])
        r_fv = float(np.corrcoef(f, v)[0, 1])
        denom = np.sqrt(max(1e-12, (1 - r_fv**2) * (1 - r_vf**2)))
        pr = (r_ff - r_fv * r_vf) / denom
        results[col] = {"raw_r": round(r_ff, 4), "partial_r_vs_vol20": round(pr, 4)}
        print(f"  {col:20s}  raw_r={r_ff:+.3f}  partial_r={pr:+.3f}")
    return results


# ── Stability check ────────────────────────────────────────────────────────────

def stability_check(X_train, y_train, X_test, y_test, feature_cols: list[str]) -> dict:
    model = VolatilityModel(model_type="ridge", horizon=HORIZON)
    model.fit(X_train[feature_cols], y_train)

    n = len(X_test)
    r1 = model.evaluate(X_test[feature_cols].iloc[:n//2], y_test.iloc[:n//2])
    r2 = model.evaluate(X_test[feature_cols].iloc[n//2:], y_test.iloc[n//2:])
    print(f"\n--- Stability (ridge, all features) ---")
    print(f"  First half:  MAE={r1.mae:.6f}  r={r1.pearson_r:+.3f}  n={n//2}")
    print(f"  Second half: MAE={r2.mae:.6f}  r={r2.pearson_r:+.3f}  n={n-n//2}")
    stable = abs(r1.pearson_r - r2.pearson_r) < 0.05
    print(f"  Stability: {'stable' if stable else 'unstable'}")
    return {
        "first_half_mae": r1.mae, "first_half_r": r1.pearson_r,
        "second_half_mae": r2.mae, "second_half_r": r2.pearson_r,
        "stable": stable,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 06: Volatility Signal v1")
    print("=" * 65)

    X_train, y_train, X_test, y_test = prepare_data()

    # Available feature columns
    all_cols    = [c for c in VOL_FEATURES if c in X_train.columns]
    vol_only    = ["vol_20"]
    ring_vol    = ["ring_radius", "vol_20"]
    phase_vol   = ["sin_theta_24h", "cos_theta_24h", "vol_20"]
    phase_ring  = ["sin_theta_24h", "cos_theta_24h", "ring_radius", "vol_20"]
    all_24h     = ["sin_theta_24h", "cos_theta_24h",
                   "sin_theta_72h", "cos_theta_72h",
                   "ring_radius", "vol_20", "lr_z20"]

    print("\n=== BASELINE: mean-only ===")
    mean_mae = float(np.mean(np.abs(y_test.values - y_test.values.mean())))
    print(f"  MAE (predict mean): {mean_mae:.6f}")

    print("\n=== BASELINES: vol_20 persistence (GARCH-0) ===")
    garch_mae = float(np.mean(np.abs(X_test["vol_20"].values - y_test.values)))
    print(f"  MAE (vol_20 = rv_12h): {garch_mae:.6f}")

    # Run model grid
    configs = [
        ("ridge", vol_only,   "RIDGE: vol_20 only"),
        ("ridge", ring_vol,   "RIDGE: ring_radius + vol_20"),
        ("ridge", phase_vol,  "RIDGE: sin/cos_24h + vol_20"),
        ("ridge", phase_ring, "RIDGE: sin/cos_24h + ring_radius + vol_20"),
        ("ridge", all_cols,   "RIDGE: all features"),
        ("rf",    phase_ring, "RF:    sin/cos_24h + ring_radius + vol_20"),
        ("rf",    all_cols,   "RF:    all features"),
    ]

    results = {}
    for mtype, cols, label in configs:
        r = run_model(X_train, y_train, X_test, y_test, mtype, cols, label)
        results[label] = {
            "model_type": mtype,
            "feature_cols": cols,
            "mae": r.mae,
            "rmse": r.rmse,
            "pearson_r": r.pearson_r,
            "mae_vs_mean": r.mae_vs_mean_baseline,
            "mae_vs_garch0": r.mae_vs_garch0_baseline,
            "partial_r_vs_vol20": r.partial_r_vs_vol20,
            "pass_garch0": r.mae_vs_garch0_baseline < 0,
        }

    # Partial correlation sweep
    partial_corrs = phase_vol_correlation_sweep(X_test, y_test)

    # Stability check (best model = phase_ring)
    stability = stability_check(X_train, y_train, X_test, y_test, phase_ring)

    # Summary
    best_label = min(results, key=lambda k: results[k]["mae"])
    best = results[best_label]
    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  GARCH-0 baseline MAE: {garch_mae:.6f}")
    print(f"  Best model:           {best_label}")
    print(f"  Best MAE:             {best['mae']:.6f}  (vs GARCH0: {best['mae_vs_garch0']:+.6f})")
    print(f"  Best Pearson r:       {best['pearson_r']:+.3f}")
    any_pass = any(r["pass_garch0"] for r in results.values())
    print(f"  H3 (any model beats GARCH0): {'PASS' if any_pass else 'FAIL'}")
    print("=" * 65)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "06_volatility_signal_v1",
        "data": str(DATA_PATH),
        "umap_train_size": TRAIN_SIZE_UMAP,
        "ml_train_size": TRAIN_SIZE_ML,
        "horizon_bars": HORIZON,
        "baselines": {"mean_mae": mean_mae, "garch0_mae": garch_mae},
        "models": results,
        "partial_correlations_vs_rv12h": partial_corrs,
        "stability": stability,
        "conclusion": {
            "best_model": best_label,
            "beats_garch0": any_pass,
            "improvement_mae": best["mae_vs_garch0"],
        },
    }
    (OUT_DIR / "volatility_v1_result.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"\nResults saved to {OUT_DIR / 'volatility_v1_result.json'}")
