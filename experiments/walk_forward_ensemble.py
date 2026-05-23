"""Ensemble walk-forward across the 5 portfolio alts + bootstrap CI on portfolio.

Compares ensemble-k=3 vs top-1 on the same 5 assets, then builds a
vol-targeted portfolio with bootstrap CI on the ensemble version.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eso.lab.walk_forward import (
    walk_forward_backtest,
    walk_forward_ensemble_backtest,
)
from eso.lab.portfolio import vol_targeted_portfolio, bootstrap_sharpe_ci


ASSET_CONFIGS = [
    ("data/VVVUSDT_4h.csv",    False),
    ("data/STGUSDT_4h.csv",    False),
    ("data/MORPHOUSDT_4h.csv", False),
    ("data/GNOUSDT_4h.csv",    True),
    ("data/GRASSUSDT_4h.csv",  False),
]


def main() -> int:
    print("=== ENSEMBLE WALK-FORWARD (k=3) — per asset ===")
    results = []
    for path, fund in ASSET_CONFIGS:
        asset = Path(path).stem
        print(f"-> {asset} ...", flush=True)
        try:
            r = walk_forward_ensemble_backtest(
                path, n_folds=4, ensemble_k=3,
                include_funding=fund,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        print(f"  Sharpe={r.aggregate_net_sharpe:+.2f}  "
              f"Net={r.aggregate_net_return*100:+.1f}%  "
              f"Trades={r.aggregate_trades}")
        results.append(r)

    if not results:
        return 1

    rep = vol_targeted_portfolio(
        results, target_annual_vol=0.30,
        output_dir="reports/portfolio_ensemble",
    )

    import pandas as pd
    curve = pd.read_csv("reports/portfolio_ensemble/portfolio_curve.csv", index_col=0)
    log_rets = curve["portfolio_log_return"].to_numpy()
    point, lo, hi = bootstrap_sharpe_ci(log_rets, bars_per_year=2190)

    print("\n=== ENSEMBLE PORTFOLIO (vol-targeted, k=3) ===")
    print(f"  Assets: {rep.assets}")
    print(f"  Avg active: {rep.avg_assets_active:.2f}")
    print(f"  Net return: {rep.portfolio_net_return*100:+.1f}%")
    print(f"  Sharpe (point): {rep.portfolio_net_sharpe:+.2f}")
    print(f"  Sharpe 95% CI:  [{lo:+.2f}, {hi:+.2f}]")
    print(f"  Max DD: {rep.portfolio_max_drawdown*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
