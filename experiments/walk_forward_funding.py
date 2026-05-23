"""Walk-forward across all alts WITH funding_rate added to feature set.

If funding rate carries genuine orthogonal information about future
returns, adding it to the UMAP feature set should change the ring
geometry — potentially producing a sharper or noisier signal than
OHLCV-only. The walk-forward will tell us which.
"""

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
    print(f"Walk-forward + funding on {len(paths)} alts...\n")
    df = walk_forward_many(
        paths,
        output_dir="reports/walkforward_funding",
        n_folds=4,
        include_funding=True,
    )
    if df.empty:
        return 1
    print("\n=== WALK-FORWARD + FUNDING RANKING ===")
    print(df.to_string(index=False))

    tradeable = df[df["aggregate_net_sharpe"] >= 1.0]
    promising = df[(df["aggregate_net_sharpe"] >= 0.5) & (df["aggregate_net_sharpe"] < 1.0)]
    print()
    print(f"TRADEABLE (Sharpe >= 1.0): {len(tradeable)}")
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
