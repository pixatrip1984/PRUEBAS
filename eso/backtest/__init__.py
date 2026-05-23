"""Cost-aware backtester for ESO causal signals.

Converts a position series and a price series into realistic PnL accounting
for fees, slippage, and turnover. Designed to be honest: no lookahead,
position at time t generates PnL from t to t+1.
"""

from eso.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from eso.backtest.strategies import (
    phase_threshold_strategy,
    proportional_strategy,
    proportional_deadband_strategy,
    regime_gated_strategy,
    gated_phase_threshold_strategy,
    long_only_baseline,
)
from eso.backtest.model_strategy import (
    model_strategy,
    build_target,
    ModelStrategyResult,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "phase_threshold_strategy",
    "long_only_baseline",
]
