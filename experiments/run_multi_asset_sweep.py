"""Run the multi-asset sweep across all alts in data/.

Usage:
    python experiments/run_multi_asset_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make eso importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.multi_asset_sweep import sweep_assets


def main() -> int:
    data_dir = Path("data")
    # All 4h alt files, excluding the BTC variants
    alt_paths = sorted([
        p for p in data_dir.glob("*USDT_4h.csv")
        if "BTC" not in p.stem
    ])
    print(f"Found {len(alt_paths)} alt assets:")
    for p in alt_paths:
        print(f"  - {p.name}")
    print()

    df = sweep_assets(
        alt_paths,
        output_dir="reports/multi_asset_sweep",
        bars_per_year=2190,
        gate_percentile=0.50,
        enter_threshold=0.3,
        exit_threshold=0.1,
    )

    if df.empty:
        print("No results.")
        return 1

    print("\n=== RANKED RESULTS (by edge per trade) ===")
    print(df[[
        "asset", "trades", "gross_sharpe", "net_sharpe",
        "net_return", "edge_bps_per_trade", "verdict"
    ]].to_string(index=False))

    # Highlight any tradeable
    tradeable = df[df["edge_bps_per_trade"] >= 20.0]
    print()
    if len(tradeable) > 0:
        print(f"*** {len(tradeable)} asset(s) with edge >= 20 bps/trade ***")
        for _, r in tradeable.iterrows():
            print(f"  {r['asset']}: {r['edge_bps_per_trade']:+.1f} bps")
    else:
        print("No assets crossed the 20 bps tradeable threshold.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
