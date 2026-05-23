"""Same paper journal with stricter quality gate (min_sharpe=1.0).

Tests whether tightening the gate would have caught the recent
underperformers (GNO, MORPHO) before they lost money.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.paper_journal import run_journal_portfolio


def main() -> int:
    summary = run_journal_portfolio(
        start_bar_offset=-200,
        output_dir="reports/paper_journal_strict",
        min_trade_size=0.05,
        min_selection_sharpe=1.0,   # ← stricter
    )
    print("\n=== PAPER JOURNAL strict (min_sharpe=1.0) ===")
    print(f"  Net return:    {summary.aggregate_net_return*100:+.1f}%")
    print(f"  Sharpe:        {summary.aggregate_sharpe:+.2f}")
    print(f"  Max DD:        {summary.aggregate_max_drawdown*100:+.1f}%")
    print(f"  Avg exposure:  {summary.avg_exposure:.2f}")
    print(f"  Trades: {summary.n_trades_per_asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
