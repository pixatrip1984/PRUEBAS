"""Walk-forward on BTC 1h (15× more bars than alts → tighter Sharpe CI).

BTC has microstructure but build_alt_feature_vector treats it as
OHLCV-only — fine, since the comparison is apples-to-apples with alts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.walk_forward import walk_forward_backtest
from eso.lab.portfolio import bootstrap_sharpe_ci


def main() -> int:
    r = walk_forward_backtest(
        "data/BTCUSDT_1h.csv",
        n_folds=6,                # more folds since we have more bars
        output_dir="reports/walkforward_btc1h",
        bars_per_year=8760,       # 1h annualisation
        fee_bps=5.5,
        slippage_bps=2.0,
    )
    print(f"\nBTC 1h walk-forward:")
    print(f"  Verdict:          {r.verdict}")
    print(f"  Aggregate trades: {r.aggregate_trades}")
    print(f"  Net return:       {r.aggregate_net_return*100:+.1f}%")
    print(f"  Net Sharpe:       {r.aggregate_net_sharpe:+.2f}")

    if r.net_returns:
        import numpy as np
        point, lo, hi = bootstrap_sharpe_ci(
            np.array(r.net_returns), bars_per_year=8760,
        )
        print(f"  Sharpe 95% CI:    [{lo:+.2f}, {hi:+.2f}]")

    print("\nPer-fold breakdown:")
    for f in r.folds:
        print(f"  fold {f['fold']}: p{int(f['selected_gate']*100)}/"
              f"{f['selected_signal']}/{f['selected_enter']:.2f} → "
              f"trades={f['trades']}  netSh={f['net_sharpe']:+.2f}  "
              f"netRet={f['net_return']*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
