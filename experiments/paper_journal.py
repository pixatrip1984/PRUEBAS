"""Run the paper-trading journal on the last 200 bars (~33 days at 4h)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.paper_journal import run_journal_portfolio


def main() -> int:
    summary = run_journal_portfolio(
        start_bar_offset=-200,  # last ~33 days at 4h
        output_dir="reports/paper_journal",
        min_trade_size=0.05,
        min_selection_sharpe=0.5,
    )
    print("\n=== PAPER-TRADING JOURNAL (last 200 bars) ===")
    print(f"  Assets:        {summary.asset_count}")
    print(f"  Steps:         {summary.n_steps}")
    print(f"  Net return:    {summary.aggregate_net_return*100:+.1f}%")
    print(f"  Sharpe:        {summary.aggregate_sharpe:+.2f}")
    print(f"  Max DD:        {summary.aggregate_max_drawdown*100:+.1f}%")
    print(f"  Avg exposure:  {summary.avg_exposure:.2f}")
    print(f"  Trades per asset: {summary.n_trades_per_asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
