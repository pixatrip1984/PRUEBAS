"""Vol-targeted portfolio with bootstrap CI and a cost stress test.

Baseline (existing): fee=5.5 bps + slippage=2 bps, equal-weight.
Stress: fee=10.0 bps + slippage=5 bps, vol-targeted.

The vol target is set so portfolio annual vol ≈ 30%, which is a
reasonable retail risk budget (1.5x BTC's typical vol).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.walk_forward import walk_forward_backtest
from eso.lab.portfolio import vol_targeted_portfolio, bootstrap_sharpe_ci


ASSET_CONFIGS = [
    ("data/VVVUSDT_4h.csv",    False),
    ("data/STGUSDT_4h.csv",    False),
    ("data/MORPHOUSDT_4h.csv", False),
    ("data/GNOUSDT_4h.csv",    True),
    ("data/GRASSUSDT_4h.csv",  False),
]


def run(label: str, fee_bps: float, slip_bps: float, out_dir: str) -> None:
    print(f"\n=== {label}  (fee={fee_bps} bps, slippage={slip_bps} bps) ===")
    results = []
    for path, fund in ASSET_CONFIGS:
        try:
            r = walk_forward_backtest(
                path, n_folds=4, include_funding=fund,
                fee_bps=fee_bps, slippage_bps=slip_bps,
            )
            results.append(r)
        except Exception as exc:
            print(f"  {Path(path).stem} failed: {exc}")

    rep = vol_targeted_portfolio(
        results, target_annual_vol=0.30,
        output_dir=out_dir,
    )

    # Bootstrap CI from raw curve
    import pandas as pd
    curve = pd.read_csv(f"{out_dir}/portfolio_curve.csv", index_col=0)
    log_rets = curve["portfolio_log_return"].to_numpy()
    point, lo, hi = bootstrap_sharpe_ci(log_rets, bars_per_year=2190)

    print(f"  Net return:    {rep.portfolio_net_return*100:+.1f}%")
    print(f"  Sharpe:        {rep.portfolio_net_sharpe:+.2f}")
    print(f"  Sharpe 95% CI: [{lo:+.2f}, {hi:+.2f}]")
    print(f"  Max DD:        {rep.portfolio_max_drawdown*100:+.1f}%")


def main() -> int:
    run(
        label="BASELINE (retail Bybit taker)",
        fee_bps=5.5, slip_bps=2.0,
        out_dir="reports/portfolio_voltarget_baseline",
    )
    run(
        label="STRESS (higher fees + slippage)",
        fee_bps=10.0, slip_bps=5.0,
        out_dir="reports/portfolio_voltarget_stress",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
