"""Reproducible benchmark: BTC geometry across timeframes and market regimes.

Runs ESO with compact financial features on BTCUSDT 1h and 3m data,
testing the S1xR2 hypothesis against circle, sphere2, torus2, cylinder, plane3d.

Usage:
    python experiments/03_btc_geometry_benchmark.py

Outputs:
    reports/benchmark_btc_geometry/  — per-window reports
    reports/benchmark_btc_geometry/summary.json  — cross-window comparison
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


MANIFOLDS = ["circle", "sphere2", "torus2", "cylinder", "s1_r2", "plane3d"]
REGISTRY = "experiments/eso_registry.csv"
BASE_OUT = Path("reports/benchmark_btc_geometry")
BASE_OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    # (label, path, klines_format, skip_rows, max_rows, dataset_id)
    ("1h_early_2021",   "data/BTCUSDT_1h.csv",              False, 0,      10000, "BTC_1h_early"),
    ("1h_mid_2022",     "data/BTCUSDT_1h.csv",              False, 18000,  10000, "BTC_1h_mid"),
    ("1h_late_2024",    "data/BTCUSDT_1h.csv",              False, 36000,  10000, "BTC_1h_late"),
    ("3m_early",        "data/btc_3m/splits/train.parquet", True,  0,      10000, "BTC_3m_early"),
    ("3m_late",         "data/btc_3m/splits/train.parquet", True,  40000,  10000, "BTC_3m_late"),
]


def run_window(label, path, klines_format, skip_rows, max_rows, dataset_id):
    out_dir = str(BASE_OUT / label)
    cmd = [
        sys.executable, "-m", "eso.cli", "explore", path,
        "--feature-mode", "compact",
        "--skip-rows", str(skip_rows),
        "--max-rows", str(max_rows),
        "--manifolds", *MANIFOLDS,
        "--k", "8",
        "--n-masks", "8",
        "--seed", "123",
        "--normalize", "robust",
        "--dataset-id", dataset_id,
        "--output", out_dir,
        "--registry", REGISTRY,
    ]
    if klines_format:
        cmd.append("--klines-format")

    print(f"\n[benchmark] Running window: {label} ({path}, skip={skip_rows}, n={max_rows})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-400:]}")
        return None

    report_path = Path(out_dir) / "report.json"
    if not report_path.exists():
        print(f"  ERROR: report.json not found at {report_path}")
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report.get("best", {})
    diag = report.get("diagnosis", {})
    dim = diag.get("dimension", {})

    row = {
        "window": label,
        "winner": best.get("manifold"),
        "reconstruction_error": round(best.get("reconstruction_error", float("nan")), 6),
        "smoothness": round(best.get("smoothness", float("nan")), 4),
        "dim_consensus": dim.get("consensus_dimension"),
        "stationary": diag.get("stationarity", {}).get("label"),
        "chaotic": diag.get("lyapunov", {}).get("chaotic_hint"),
        "full_ranking": [
            {
                "manifold": e["manifold"],
                "error": round(e["reconstruction_error"], 6),
                "smoothness": round(e["smoothness"], 4),
            }
            for e in sorted(report.get("evaluations", []), key=lambda x: x["reconstruction_error"])
        ],
    }
    print(f"  winner={row['winner']}  error={row['reconstruction_error']}  dim={row['dim_consensus']}  stationary={row['stationary']}")
    return row


def main():
    results = []
    for window_args in WINDOWS:
        row = run_window(*window_args)
        if row:
            results.append(row)

    summary_path = BASE_OUT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[benchmark] Summary written to {summary_path}")

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'Window':<20} {'Winner':<10} {'Error':<10} {'Smooth':<8} {'dim':<5} {'Stationary'}")
    print("-" * 68)
    for r in results:
        print(f"{r['window']:<20} {r['winner']:<10} {r['reconstruction_error']:<10} {r['smoothness']:<8} {r['dim_consensus']:<5} {r['stationary']}")

    winners = [r["winner"] for r in results if r.get("winner")]
    dominant = max(set(winners), key=winners.count) if winners else "unknown"
    stable = all(w == dominant for w in winners)
    print(f"\nDominant geometry: {dominant}")
    print(f"Stable across all windows: {stable}")


if __name__ == "__main__":
    main()
