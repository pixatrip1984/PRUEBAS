"""Heterogeneous portfolio: each asset with its best feature configuration.

Walk-forward result per asset chose its own best (gate, signal, enter)
per fold. Now we also choose per-asset whether to include funding_rate.
Then equal-weight portfolio across all winners.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.walk_forward import walk_forward_backtest
from eso.lab.portfolio import equal_weight_portfolio


# Per-asset best feature configuration based on the walk-forward sweep
# format: (csv_path, include_funding)
ASSET_CONFIGS = [
    ("data/VVVUSDT_4h.csv",    False),  # better w/o funding (Sharpe 3.72 vs 2.98)
    ("data/STGUSDT_4h.csv",    False),  # better w/o funding (Sharpe 1.72 vs 0.65)
    ("data/MORPHOUSDT_4h.csv", False),  # better w/o funding (Sharpe 1.40 vs -0.48)
    ("data/GNOUSDT_4h.csv",    True),   # only tradeable WITH funding (Sharpe 1.55 vs -0.74)
    ("data/GRASSUSDT_4h.csv",  False),  # better w/o funding (Sharpe 0.58 vs -0.50)
]


def main() -> int:
    results = []
    for path, fund in ASSET_CONFIGS:
        asset = Path(path).stem
        tag = "+funding" if fund else "OHLCV-only"
        print(f"-> {asset} ({tag}) ...", flush=True)
        try:
            r = walk_forward_backtest(path, n_folds=4, include_funding=fund)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        print(f"  Sharpe={r.aggregate_net_sharpe:+.2f}  "
              f"Net={r.aggregate_net_return*100:+.1f}%")
        results.append(r)

    if not results:
        return 1

    rep = equal_weight_portfolio(
        results, bars_per_year=2190,
        output_dir="reports/portfolio_heterogeneous",
    )
    print("\n=== HETEROGENEOUS PORTFOLIO (per-asset best features) ===")
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
