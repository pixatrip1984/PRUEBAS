"""Build the equal-weight portfolio of the 4 walk-forward-tradeable alts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.portfolio import run_portfolio


TRADEABLE_ALTS = [
    "data/VVVUSDT_4h.csv",
    "data/STGUSDT_4h.csv",
    "data/MORPHOUSDT_4h.csv",
    "data/GRASSUSDT_4h.csv",
]


def main() -> int:
    rep = run_portfolio(TRADEABLE_ALTS, output_dir="reports/portfolio_walkforward")
    print("\n=== EQUAL-WEIGHT PORTFOLIO (walk-forward) ===")
    print(f"  Assets: {rep.assets}")
    print(f"  Bars (union): {rep.n_bars_union}")
    print(f"  Bars (intersection): {rep.n_bars_intersection}")
    print(f"  Avg assets active: {rep.avg_assets_active:.2f}")
    print(f"  Net return: {rep.portfolio_net_return*100:+.1f}%")
    print(f"  Net Sharpe: {rep.portfolio_net_sharpe:+.2f}")
    print(f"  Max DD: {rep.portfolio_max_drawdown*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
