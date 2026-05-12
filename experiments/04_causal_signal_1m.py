"""Experiment 04: Causal phase signal on 1-minute BTC data.

Data: data/btc_3m/splits/train.parquet
      92,736 rows of 1-MINUTE bars (Binance Klines), Oct-Dec 2024.
      Note: folder is named "btc_3m" but bars are 1-minute (60s between rows).

Hypothesis:
    The S1 ring found in 1h compact features also exists in 1-min compact features.
    Causal phase signal achieves r_OOS >= 0.15 at +720 bars (= 12h calendar).

Findings (run 2026-05-12):
    Resolution threshold for S1 ring: ~15-30 minutes.
    - 1-min, 5-min: ring_cv > 0.49 — NO RING (microstructure noise dominates)
    - 15-min: ring_cv=0.29 (transductive), causal |r_OOS|=0.292
    - 30-min: ring_cv=0.249, causal |r_OOS|=0.303 (ring holds, sign inverted)
    - 1h:     ring_cv=0.228, |r_OOS|=0.320 (reference)
    Verdict: PARTIAL PASS — ring threshold at ~15min confirmed.
    1-min specific: WEAK (r_OOS=0.055, no ring). Use 15-min+ for sub-hourly signals.

Reference: 1h result = r_OOS=0.320 at +12 bars (= 12h calendar), train_size=27000.

Method:
    CausalCyclePhase(train_size=55000) — fit UMAP on first ~38 days,
    evaluate on remaining ~26 days (all Oct-Dec 2024 in OOS).
    Smooth windows tested: 60, 360, 720, 1440 bars (= 1h, 6h, 12h, 24h calendar).
    Primary horizon: 720 bars (= 12h calendar equivalent).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd

from eso.data.features import prepare_btc_klines, build_financial_features
from eso.signals.causal_cycle import CausalCyclePhase
from eso.signals.cycle import ring_quality


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH   = pathlib.Path("data/btc_3m/splits/train.parquet")
OUT_DIR     = pathlib.Path("reports/causal_1m")
TRAIN_SIZE  = 55_000
SEED        = 42
# Primary horizon (bars). 720 bars × 1 min/bar = 720 min = 12h calendar.
PRIMARY_HORIZON = 720
# Smooth windows in bars: 1h=60, 6h=360, 12h=720, 24h=1440
SMOOTH_WINDOWS = (60, 360, 720, 1440)


# ── Load and prepare ───────────────────────────────────────────────────────────

def load_1m_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = prepare_btc_klines(df)
    # Add timestamp column (ISO string) from open_datetime
    if "open_datetime" in df.columns:
        df["timestamp"] = df["open_datetime"].astype(str)
    elif "open_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).astype(str)
    df = df.reset_index(drop=True)
    print(f"Loaded: {len(df):,} rows  |  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"Columns: {list(df.columns)}")
    return df


# ── Quick ring quality check ───────────────────────────────────────────────────

def check_ring(df: pd.DataFrame, n_rows: int = 5000) -> dict:
    """Quick transductive ring check on last n_rows (using cycle.py)."""
    from eso.signals.cycle import extract_cycle_phase
    sample = df.tail(n_rows).reset_index(drop=True)
    phase = extract_cycle_phase(sample, seed=SEED)
    q = ring_quality(phase)
    print(f"\n--- Ring quality (transductive, last {n_rows} bars) ---")
    print(f"  radius_cv     = {q['radius_cv']:.3f}  (< 0.3 = ring; 1h result = 0.279)")
    print(f"  angle_uniform = {q['angle_uniformity']:.3f}")
    print(f"  ring_score    = {q['ring_score']:.3f}")
    print(f"  is_ring       = {q['is_ring']}")
    return q


# ── Causal evaluation ──────────────────────────────────────────────────────────

def run_causal_evaluation(df: pd.DataFrame) -> dict:
    """Causal split: fit UMAP on first TRAIN_SIZE bars, evaluate on rest."""
    print(f"\n--- Causal evaluation ---")
    print(f"  train_size = {TRAIN_SIZE:,} bars  (~{TRAIN_SIZE/60/24:.1f} days)")

    causal = CausalCyclePhase(
        train_size=TRAIN_SIZE,
        n_neighbors=15,
        min_dist=0.1,
        seed=SEED,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = causal.fit_and_evaluate(
            df,
            horizon_h=PRIMARY_HORIZON,
            smooth_windows=SMOOTH_WINDOWS,
        )

    print(f"\n  Train bars : {result.train_size:,}")
    print(f"  Test bars  : {result.test_size:,}  (~{result.test_size/60/24:.1f} days)")
    print(f"  Train ring_cv : {result.train_ring_cv:.3f}")
    print(f"  Test  ring_cv : {result.test_ring_cv:.3f}  (is_ring={result.test_is_ring})")
    print(f"\n  Correlations at horizon={PRIMARY_HORIZON} bars (= {PRIMARY_HORIZON}min = {PRIMARY_HORIZON/60:.1f}h):")
    for smooth, corrs in sorted(result.correlations.items()):
        for h, r in sorted(corrs.items()):
            if h == PRIMARY_HORIZON:
                print(f"    smooth={smooth:>5}bars ({smooth/60:4.1f}h)  r_OOS={r:+.3f}")

    print(f"\n  Best overall: smooth={result.best_smooth}  horizon={result.best_horizon}  r_OOS={result.best_r:+.3f}")
    print(f"  Reference 1h: smooth=24  horizon=12  r_OOS=+0.320")

    return result


# ── Horizon sweep ──────────────────────────────────────────────────────────────

def horizon_sweep(df: pd.DataFrame, smooth: int = 720, horizons: tuple = (60, 180, 360, 720, 1440)) -> dict:
    """Sweep horizons at a fixed smooth window to find peak predictability."""
    print(f"\n--- Horizon sweep (smooth={smooth} bars = {smooth/60:.1f}h) ---")
    from eso.data.features import build_financial_features, feature_column_groups
    feat = build_financial_features(df).dropna()
    groups = feature_column_groups()
    cols = [c for c in groups["compact"] if c in feat.columns]
    feat_clean = feat[cols + ["log_return"]]
    n = len(feat_clean)
    train_n = min(TRAIN_SIZE, int(n * 0.6))

    # Fit causal UMAP once
    causal = CausalCyclePhase(train_size=train_n, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causal.fit(feat_clean[cols].iloc[:train_n])
        emb = causal.transform(feat_clean[cols].iloc[train_n:])

    theta = np.unwrap(np.arctan2(emb[:, 1], emb[:, 0]))

    # Smooth phase at selected window
    z_re = pd.Series(np.cos(theta)).rolling(smooth, min_periods=max(1, smooth // 4), center=False).mean().to_numpy()
    z_im = pd.Series(np.sin(theta)).rolling(smooth, min_periods=max(1, smooth // 4), center=False).mean().to_numpy()
    theta_s = np.angle(z_re + 1j * z_im)
    sin_s = np.sin(theta_s)
    cos_s = np.cos(theta_s)

    lr_test = feat.loc[feat_clean.index[train_n:], "log_return"].to_numpy()

    results = {}
    for h in horizons:
        if h >= len(sin_s):
            continue
        sin_lag = sin_s[:-h]
        cos_lag = cos_s[:-h]
        lr_fwd  = lr_test[h:]
        n_pair  = min(len(sin_lag), len(lr_fwd))
        if n_pair < 100:
            continue
        r_sin = np.corrcoef(sin_lag[:n_pair], lr_fwd[:n_pair])[0, 1]
        r_cos = np.corrcoef(cos_lag[:n_pair], lr_fwd[:n_pair])[0, 1]
        r_best = max(abs(r_sin), abs(r_cos)) * np.sign(r_sin if abs(r_sin) >= abs(r_cos) else r_cos)
        results[h] = float(r_best)
        cal_h = h / 60
        print(f"  h={h:>5} bars ({cal_h:5.1f}h)  r={r_best:+.3f}")

    return results


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(ring_q: dict, result, sweep: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "04_causal_signal_1m",
        "data": str(DATA_PATH),
        "bar_resolution": "1-minute",
        "n_rows": 92_736,
        "date_range": "2024-10-01 to 2024-12-04",
        "train_size": result.train_size,
        "test_size": result.test_size,
        "ring_quality_transductive": ring_q,
        "causal_ring_cv_train": result.train_ring_cv,
        "causal_ring_cv_test": result.test_ring_cv,
        "causal_is_ring_test": result.test_is_ring,
        "best_r_oos": result.best_r,
        "best_smooth_bars": result.best_smooth,
        "best_horizon_bars": result.best_horizon,
        "best_smooth_hours": result.best_smooth / 60,
        "best_horizon_hours": result.best_horizon / 60,
        "correlations": {str(k): v for k, v in result.correlations.items()},
        "horizon_sweep_smooth720": sweep,
        "reference_1h": {
            "r_oos": 0.320,
            "smooth_bars": 24,
            "horizon_bars": 12,
            "smooth_hours": 24,
            "horizon_hours": 12,
        },
    }

    out_file = OUT_DIR / "causal_1m_result.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {out_file}")

    # Also save test phase CSV
    result.test_phase.to_csv(OUT_DIR / "test_phase_1m.csv")
    print(f"Test phase saved to {OUT_DIR / 'test_phase_1m.csv'}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 04: Causal S¹ signal on 1-minute BTC data")
    print("=" * 60)

    df = load_1m_data()

    # Step 1: Quick ring check (transductive, to verify S¹ exists at all)
    ring_q = check_ring(df)

    # Step 2: Full causal evaluation
    result = run_causal_evaluation(df)

    # Step 3: Horizon sweep at best smooth window
    best_smooth = result.best_smooth if result.best_smooth else 720
    sweep = horizon_sweep(df, smooth=best_smooth)

    # Step 4: Save
    save_results(ring_q, result, sweep)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print(f"  Ring exists (causal test): {result.test_is_ring}  (CV={result.test_ring_cv:.3f})")
    print(f"  Best r_OOS = {result.best_r:+.3f}  at smooth={result.best_smooth}  h={result.best_horizon}")
    print(f"  Reference 1h r_OOS = +0.320")
    verdict = "PASS" if result.best_r >= 0.15 else ("WEAK" if result.best_r >= 0.05 else "FAIL")
    print(f"  Verdict: {verdict}  (target >= 0.15)")
    print("=" * 60)
