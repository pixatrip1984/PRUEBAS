"""ESO signals — structural features and causal targets."""

from .targets import (
    build_market_target_frame,
    future_direction,
    future_log_return,
    future_realized_volatility,
    regime_from_training_median,
)

__all__ = [
    "build_market_target_frame",
    "future_direction",
    "future_log_return",
    "future_realized_volatility",
    "regime_from_training_median",
]
