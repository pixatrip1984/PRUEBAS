"""Experiment 08: Conditional direction model — cycle phase in low-vol regimes.

Key finding from exp 07 analysis: the direction signal from cycle phase
exists ONLY in low-volatility regimes.

Causal correlations (center=False, OOS):
  All regimes:  r(sin_theta_24h, future_12h) = -0.042  (no signal)
  Vol < Q25:    r(sin_theta_24h, future_12h) = -0.133  (signal!)
  Vol < Q10:    r(sin_theta_24h, future_12h) = -0.199  (strong signal!)

Hypothesis: A logistic regression on [sin_theta_24h, cos_theta_24h, vol_20]
evaluated ONLY in low-vol regime (vol_20 < Q25 of training set) achieves
accuracy > 55% OOS on sign(log_return(t+12)).

Methodology:
  - Use the realized_vol_12h from the vol model to define "low-vol" bar
  - Alternatively: use vol_20 < percentile threshold (purely causal at time t)
  - Only issue predictions in low-vol windows; no prediction in high-vol

Train: first 15,000 OOS rows  | Test: last 4,474 OOS rows
Causal at time t: all features computed from data up to t only.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from eso.signals.feature_vector import build_feature_vector


OUT_DIR         = pathlib.Path("reports/conditional_direction")
DATA_PATH       = pathlib.Path("data/BTCUSDT_1h.csv")
TRAIN_SIZE_UMAP = 27_000
TRAIN_SIZE_ML   = 15_000
HORIZON         = 12


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fv = build_feature_vector(df, train_size=TRAIN_SIZE_UMAP, smooth_windows=(6, 24, 72))

    close = fv["close"].values
    n = len(fv)

    # Causal future log_return (no lookahead)
    future_lr12 = np.full(n, np.nan)
    future_lr12[:n-HORIZON] = np.log(close[HORIZON:] / close[:n-HORIZON])

    fv2 = fv.copy()
    fv2["future_lr12"] = future_lr12
    fv2["target"] = np.sign(fv2["future_lr12"])
    fv2 = fv2.dropna(subset=["future_lr12"]).reset_index(drop=True)
    fv2 = fv2[fv2["target"] != 0]

    print(f"Full dataset: {len(fv2):,} rows")
    print(f"Date range: {fv2['timestamp'].iloc[0]} -> {fv2['timestamp'].iloc[-1]}")
    return fv2


def run_conditional_model(fv, vol_percentile, label):
    """Train and evaluate on low-vol subset of OOS data."""
    train = fv.iloc[:TRAIN_SIZE_ML]
    test  = fv.iloc[TRAIN_SIZE_ML:]

    # Vol threshold from training data (causal: use vol_20 at time t)
    vol_threshold = float(np.nanpercentile(train["vol_20"].values, vol_percentile))

    # Train: all data (model must learn the pattern across vol levels, then apply to low-vol)
    feat_cols = ["sin_theta_24h", "cos_theta_24h", "ring_radius", "vol_20", "lr_z20"]
    feat_cols = [c for c in feat_cols if c in fv.columns]

    X_train = train[feat_cols]
    y_train = train["target"].astype(int)

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.to_numpy(dtype=float))

    model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train_s, y_train)

    # Test: evaluate ONLY on low-vol bars
    test_lv = test[test["vol_20"] < vol_threshold]

    if len(test_lv) < 50:
        print(f"\n{label}: insufficient test samples ({len(test_lv)})")
        return None

    X_test_s = scaler.transform(test_lv[feat_cols].to_numpy(dtype=float))
    y_pred = model.predict(X_test_s)
    y_true = test_lv["target"].astype(int).values

    acc = float((y_pred == y_true).mean())
    majority = float((y_true == 1).mean())
    baseline = max(majority, 1 - majority)
    coverage = len(test_lv) / len(test)

    verdict = "PASS" if acc > 0.55 else ("MARGINAL" if acc > 0.52 else "FAIL")
    print(f"\n{label}")
    print(f"  vol_threshold:  {vol_threshold:.5f}  (Q{vol_percentile} of training vol_20)")
    print(f"  Test coverage:  {len(test_lv)}/{len(test)} ({coverage*100:.1f}% of bars)")
    print(f"  Test balance:   +1={majority:.3f}  -1={1-majority:.3f}")
    print(f"  Accuracy:       {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Baseline:       {baseline:.4f}  (majority class)")
    print(f"  vs Baseline:    {acc-baseline:+.4f}  ({(acc-baseline)*100:+.1f}pp)")
    print(f"  Verdict: {verdict}  (target > 55%)")

    return {
        "vol_percentile": vol_percentile,
        "vol_threshold": vol_threshold,
        "n_test_total": len(test),
        "n_test_lv": len(test_lv),
        "coverage": coverage,
        "accuracy": acc,
        "baseline": baseline,
        "vs_baseline": acc - baseline,
        "pass": acc > 0.55,
        "verdict": verdict,
    }


def correlation_profile(fv) -> dict:
    """Direction correlation as a function of vol percentile threshold."""
    print("\n--- Conditional correlation profile ---")
    print("  (r(sin_theta_24h, future_12h) in subsets below each percentile)")
    results = {}
    s24 = fv["sin_theta_24h"].values
    fut = fv["future_lr12"].values
    v   = fv["vol_20"].values
    mask = ~np.isnan(fut)

    for pct in [10, 15, 20, 25, 33, 50, 75, 100]:
        thresh = np.nanpercentile(v, pct)
        m = mask & (v < thresh)
        if m.sum() < 100:
            continue
        r = float(np.corrcoef(s24[m], fut[m])[0, 1])
        print(f"  vol < Q{pct:3d} ({thresh:.5f}):  r={r:+.4f}  n={m.sum()}")
        results[f"Q{pct}"] = {"threshold": thresh, "r": r, "n": int(m.sum())}
    return results


if __name__ == "__main__":
    print("=" * 65)
    print("Experiment 08: Conditional Direction Model")
    print("=" * 65)

    fv = prepare_data()

    # Correlation profile
    corr_profile = correlation_profile(fv)

    # Conditional model at multiple vol thresholds
    print("\n=== CONDITIONAL DIRECTION MODELS ===")
    results = {}
    for pct, label in [
        (10, "vol < Q10 (very low vol)"),
        (20, "vol < Q20"),
        (25, "vol < Q25"),
        (33, "vol < Q33"),
        (50, "vol < Q50 (half data)"),
    ]:
        r = run_conditional_model(fv, pct, label)
        if r:
            results[label] = r

    # Stability check (best vol threshold)
    print("\n=== STABILITY CHECK (vol < Q25) ===")
    test = fv.iloc[TRAIN_SIZE_ML:].copy()
    train = fv.iloc[:TRAIN_SIZE_ML].copy()
    vol_q25 = float(np.nanpercentile(train["vol_20"].values, 25))
    test_lv = test[test["vol_20"] < vol_q25].reset_index(drop=True)
    n_lv = len(test_lv)
    if n_lv > 100:
        h1 = test_lv.iloc[:n_lv//2]
        h2 = test_lv.iloc[n_lv//2:]

        feat_cols = [c for c in ["sin_theta_24h","cos_theta_24h","ring_radius","vol_20","lr_z20"]
                     if c in fv.columns]
        scaler = RobustScaler().fit(train[feat_cols].to_numpy(dtype=float))
        model  = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
        model.fit(scaler.transform(train[feat_cols].to_numpy(dtype=float)),
                  train["target"].astype(int))

        acc1 = float((model.predict(scaler.transform(h1[feat_cols].to_numpy(dtype=float)))
                      == h1["target"].astype(int).values).mean())
        acc2 = float((model.predict(scaler.transform(h2[feat_cols].to_numpy(dtype=float)))
                      == h2["target"].astype(int).values).mean())
        print(f"  First half:  {acc1:.4f}  n={len(h1)}")
        print(f"  Second half: {acc2:.4f}  n={len(h2)}")
        stability = {"first_half": acc1, "second_half": acc2,
                     "stable": abs(acc1-acc2) < 0.03}
    else:
        stability = {}

    # Summary
    passing = [k for k, v in results.items() if v.get("pass")]
    best    = max(results.items(), key=lambda x: x[1]["accuracy"])
    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  Passing models (>55%): {len(passing)}/{len(results)}")
    for k in passing:
        print(f"    {k}: {results[k]['accuracy']:.4f} (n={results[k]['n_test_lv']})")
    print(f"  Best: {best[0]} -> acc={best[1]['accuracy']:.4f}")
    any_pass = bool(passing)
    print(f"  H1 (any conditional model > 55%): {'PASS' if any_pass else 'FAIL'}")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "conditional_direction_result.json").write_text(json.dumps({
        "experiment": "08_conditional_direction",
        "data": str(DATA_PATH), "umap_train_size": TRAIN_SIZE_UMAP,
        "ml_train_size": TRAIN_SIZE_ML, "horizon_bars": HORIZON,
        "correlation_profile": corr_profile,
        "models": results,
        "stability": stability,
        "conclusion": {
            "any_model_passes": any_pass,
            "best_model": best[0] if results else None,
            "best_accuracy": best[1]["accuracy"] if results else None,
        },
    }, indent=2, default=str))
    print(f"\nResults saved to {OUT_DIR}")
