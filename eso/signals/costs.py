"""Trading cost sanity checks for market-signal research.

All edge and cost inputs are decimal rates: 0.0013 means 0.13%.
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict


CostDecision = Literal["PASS", "WATCH", "FAIL"]

DEFAULT_TAKER_ROUND_TRIP_COST = 0.0013
DEFAULT_MAKER_ROUND_TRIP_REBATE = 0.0002
DEFAULT_WATCH_BUFFER = 0.0002


class CostCheckResult(TypedDict):
    decision: CostDecision
    gross_edge: float
    cost_floor: float
    net_edge: float
    watch_buffer: float
    passes_floor: bool
    gross_edge_bps: float
    cost_floor_bps: float
    net_edge_bps: float
    explanation: str


def bps_to_rate(bps: float) -> float:
    """Convert basis points to a decimal rate."""
    bps = _finite_float(bps, "bps")
    return bps / 10000.0


def rate_to_bps(rate: float) -> float:
    """Convert a decimal rate to basis points."""
    rate = _finite_float(rate, "rate")
    return rate * 10000.0


def round_trip_cost_floor(
    taker_round_trip_cost: float = DEFAULT_TAKER_ROUND_TRIP_COST,
    slippage_round_trip: float = 0.0,
    maker_round_trip_rebate: float = 0.0,
) -> float:
    """Return the effective round-trip cost floor as a decimal rate.

    For maker-only assumptions, pass taker_round_trip_cost=0.0 and a positive
    maker_round_trip_rebate. The result may be negative when rebates exceed
    explicit costs.
    """
    taker_round_trip_cost = _non_negative_float(
        taker_round_trip_cost,
        "taker_round_trip_cost",
    )
    slippage_round_trip = _non_negative_float(
        slippage_round_trip,
        "slippage_round_trip",
    )
    maker_round_trip_rebate = _non_negative_float(
        maker_round_trip_rebate,
        "maker_round_trip_rebate",
    )
    return taker_round_trip_cost + slippage_round_trip - maker_round_trip_rebate


def net_edge_after_costs(
    gross_edge: float,
    taker_round_trip_cost: float = DEFAULT_TAKER_ROUND_TRIP_COST,
    slippage_round_trip: float = 0.0,
    maker_round_trip_rebate: float = 0.0,
) -> float:
    """Return gross edge minus the effective round-trip cost floor."""
    gross_edge = _finite_float(gross_edge, "gross_edge")
    return gross_edge - round_trip_cost_floor(
        taker_round_trip_cost=taker_round_trip_cost,
        slippage_round_trip=slippage_round_trip,
        maker_round_trip_rebate=maker_round_trip_rebate,
    )


def evaluate_cost_floor(
    gross_edge: float,
    taker_round_trip_cost: float = DEFAULT_TAKER_ROUND_TRIP_COST,
    slippage_round_trip: float = 0.0,
    maker_round_trip_rebate: float = 0.0,
    watch_buffer: float = DEFAULT_WATCH_BUFFER,
) -> CostCheckResult:
    """Classify a gross per-trade edge against trading costs.

    Decisions:
    - FAIL: gross edge does not clear the cost floor.
    - WATCH: edge clears the floor, but the net buffer is thin.
    - PASS: edge clears the floor by at least watch_buffer.
    """
    gross_edge = _finite_float(gross_edge, "gross_edge")
    watch_buffer = _non_negative_float(watch_buffer, "watch_buffer")
    cost_floor = round_trip_cost_floor(
        taker_round_trip_cost=taker_round_trip_cost,
        slippage_round_trip=slippage_round_trip,
        maker_round_trip_rebate=maker_round_trip_rebate,
    )
    net_edge = gross_edge - cost_floor

    if net_edge <= 0.0:
        decision: CostDecision = "FAIL"
        explanation = "gross edge does not clear the round-trip cost floor"
    elif net_edge < watch_buffer:
        decision = "WATCH"
        explanation = "gross edge clears costs, but net buffer is thin"
    else:
        decision = "PASS"
        explanation = "gross edge clears costs with usable buffer"

    return {
        "decision": decision,
        "gross_edge": gross_edge,
        "cost_floor": cost_floor,
        "net_edge": net_edge,
        "watch_buffer": watch_buffer,
        "passes_floor": net_edge > 0.0,
        "gross_edge_bps": rate_to_bps(gross_edge),
        "cost_floor_bps": rate_to_bps(cost_floor),
        "net_edge_bps": rate_to_bps(net_edge),
        "explanation": explanation,
    }


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _non_negative_float(value: float, name: str) -> float:
    out = _finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return out

