"""Walk-forward on every alt — find which ones survive forward selection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.walk_forward import walk_forward_many


def main() -> int:
    data_dir = Path("data")
    paths = sorted([
        p for p in data_dir.glob("*USDT_4h.csv")
        if "BTC" not in p.stem
    ])
    print(f"Walk-forward on {len(paths)} alts...\n")
    df = walk_forward_many(paths, output_dir="reports/walkforward_all", n_folds=4)
    if df.empty:
        print("No results.")
        return 1

    print("\n=== WALK-FORWARD RANKING (all alts) ===")
    print(df.to_string(index=False))

    tradeable = df[df["aggregate_net_sharpe"] >= 1.0]
    promising = df[(df["aggregate_net_sharpe"] >= 0.5) & (df["aggregate_net_sharpe"] < 1.0)]
    print()
    print(f"TRADEABLE (net Sharpe >= 1.0): {len(tradeable)}")
    for _, r in tradeable.iterrows():
        print(f"  {r['asset']:18s} Sharpe={r['aggregate_net_sharpe']:+.2f}  "
              f"Net={r['aggregate_net_return']*100:+.1f}%")
    print(f"PROMISING (0.5 <= Sharpe < 1.0): {len(promising)}")
    for _, r in promising.iterrows():
        print(f"  {r['asset']:18s} Sharpe={r['aggregate_net_sharpe']:+.2f}  "
              f"Net={r['aggregate_net_return']*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
