import pytest

from eso.cli import main
from eso.signals.costs import (
    DEFAULT_MAKER_ROUND_TRIP_REBATE,
    DEFAULT_TAKER_ROUND_TRIP_COST,
    bps_to_rate,
    evaluate_cost_floor,
    net_edge_after_costs,
    rate_to_bps,
    round_trip_cost_floor,
)


def test_default_floor_matches_taker_round_trip_cost():
    assert round_trip_cost_floor() == pytest.approx(0.0013)
    assert round_trip_cost_floor() == pytest.approx(DEFAULT_TAKER_ROUND_TRIP_COST)
    assert rate_to_bps(DEFAULT_TAKER_ROUND_TRIP_COST) == pytest.approx(13.0)


def test_fail_when_gross_edge_does_not_clear_cost_floor():
    result = evaluate_cost_floor(0.0012)

    assert result["decision"] == "FAIL"
    assert result["passes_floor"] is False
    assert result["cost_floor"] == pytest.approx(0.0013)
    assert result["net_edge"] == pytest.approx(-0.0001)
    assert "does not clear" in result["explanation"]


def test_watch_when_edge_only_barely_clears_floor():
    result = evaluate_cost_floor(0.0014)

    assert result["decision"] == "WATCH"
    assert result["passes_floor"] is True
    assert result["net_edge"] == pytest.approx(0.0001)
    assert result["net_edge_bps"] == pytest.approx(1.0)


def test_pass_when_edge_clears_floor_with_buffer():
    result = evaluate_cost_floor(0.0016)

    assert result["decision"] == "PASS"
    assert result["passes_floor"] is True
    assert result["net_edge"] == pytest.approx(0.0003)
    assert "usable buffer" in result["explanation"]


def test_maker_rebate_can_make_small_edge_viable():
    result = evaluate_cost_floor(
        0.0001,
        taker_round_trip_cost=0.0,
        maker_round_trip_rebate=DEFAULT_MAKER_ROUND_TRIP_REBATE,
    )

    assert result["decision"] == "PASS"
    assert result["cost_floor"] == pytest.approx(-0.0002)
    assert result["net_edge"] == pytest.approx(0.0003)


def test_slippage_adds_to_cost_floor_and_net_edge():
    floor = round_trip_cost_floor(slippage_round_trip=0.0004)

    assert floor == pytest.approx(0.0017)
    assert net_edge_after_costs(0.0020, slippage_round_trip=0.0004) == pytest.approx(0.0003)


def test_basis_point_helpers_are_inverse_for_common_values():
    assert bps_to_rate(13.0) == pytest.approx(0.0013)
    assert rate_to_bps(bps_to_rate(2.5)) == pytest.approx(2.5)


def test_rejects_non_finite_and_negative_cost_inputs():
    with pytest.raises(ValueError, match="gross_edge"):
        evaluate_cost_floor(float("nan"))
    with pytest.raises(ValueError, match="slippage_round_trip"):
        round_trip_cost_floor(slippage_round_trip=-0.0001)
    with pytest.raises(ValueError, match="watch_buffer"):
        evaluate_cost_floor(0.0020, watch_buffer=-0.0001)


def test_cli_cost_check_returns_nonzero_only_for_fail(capsys):
    assert main(["cost-check", "--gross-edge-bps", "14"]) == 0
    out = capsys.readouterr().out
    assert '"decision": "WATCH"' in out

    assert main(["cost-check", "--gross-edge-bps", "12"]) == 1
    out = capsys.readouterr().out
    assert '"decision": "FAIL"' in out
