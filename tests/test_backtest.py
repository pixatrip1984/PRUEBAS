"""Tests for cost-aware backtester and baseline strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eso.backtest import (
    BacktestConfig,
    run_backtest,
    phase_threshold_strategy,
    proportional_strategy,
    proportional_deadband_strategy,
    regime_gated_strategy,
    long_only_baseline,
    model_strategy,
    build_target,
)


def _make_prices(returns: list[float]) -> pd.Series:
    """Build a price series from log-returns starting at 100."""
    log_prices = np.log(100.0) + np.cumsum([0.0] + returns)
    return pd.Series(np.exp(log_prices), name="close")


def test_zero_position_yields_no_pnl_no_cost():
    prices = _make_prices([0.01, -0.005, 0.02, 0.0])
    positions = pd.Series([0.0] * len(prices), index=prices.index)
    r = run_backtest(prices, positions)
    assert r.metrics["gross_total_return"] == pytest.approx(0.0)
    assert r.metrics["net_total_return"] == pytest.approx(0.0)
    assert r.metrics["n_trades"] == 0
    assert r.metrics["cost_drag"] == pytest.approx(0.0)


def test_long_only_matches_buy_and_hold_minus_entry_cost():
    prices = _make_prices([0.01, -0.005, 0.02, 0.01])
    positions = pd.Series([1.0] * len(prices), index=prices.index)
    cfg = BacktestConfig(fee_bps=5.5, slippage_bps=2.0)
    r = run_backtest(prices, positions, cfg)
    expected_gross = float(prices.iloc[-1] / prices.iloc[0] - 1)
    assert r.metrics["gross_total_return"] == pytest.approx(expected_gross, rel=1e-3)
    # Exactly one entry of size 1 → one leg of cost
    assert r.metrics["n_trades"] == 1
    # Cost drag in arithmetic-return space ≈ cost_in_log_space * e^gross_total
    expected_cost_drag = cfg.cost_per_unit_turnover * (1.0 + expected_gross)
    assert r.metrics["cost_drag"] == pytest.approx(expected_cost_drag, rel=5e-2)


def test_short_position_inverts_pnl():
    prices = _make_prices([0.01, 0.01, 0.01, 0.01])  # monotone up
    long_pos = pd.Series([1.0] * len(prices), index=prices.index)
    short_pos = pd.Series([-1.0] * len(prices), index=prices.index)
    r_long = run_backtest(prices, long_pos)
    r_short = run_backtest(prices, short_pos)
    # Log returns are perfectly anti-symmetric; total return signs must flip.
    assert r_long.metrics["gross_total_return"] > 0
    assert r_short.metrics["gross_total_return"] < 0


def test_flip_increases_turnover_and_cost():
    prices = _make_prices([0.01, -0.005, 0.02, 0.0, 0.01])
    flipping = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], index=prices.index)
    holding = pd.Series([1.0] * len(prices), index=prices.index)
    cfg = BacktestConfig(fee_bps=5.5, slippage_bps=2.0)
    r_flip = run_backtest(prices, flipping, cfg)
    r_hold = run_backtest(prices, holding, cfg)
    assert r_flip.metrics["n_trades"] > r_hold.metrics["n_trades"]
    assert r_flip.metrics["cost_drag"] > r_hold.metrics["cost_drag"]


def test_disallow_short_clips_positions():
    prices = _make_prices([0.01, -0.005, 0.02])
    positions = pd.Series([-1.0] * len(prices), index=prices.index)
    cfg = BacktestConfig(allow_short=False)
    r = run_backtest(prices, positions, cfg)
    # Positions clipped to >=0, so no PnL at all
    assert r.metrics["gross_total_return"] == pytest.approx(0.0)


def test_phase_threshold_strategy_respects_hysteresis():
    # Signal oscillates: stays above enter, then drops between exit and enter
    sig = pd.Series([0.0, 0.4, 0.5, 0.2, 0.05, -0.4, -0.5, -0.2, -0.05, 0.0])
    fv = pd.DataFrame({"cos_theta_24h": sig})
    pos = phase_threshold_strategy(
        fv, signal_col="cos_theta_24h",
        enter_threshold=0.3, exit_threshold=0.1,
        allow_short=True,
    )
    # Bar 1: enters long, holds through bar 3 (0.2 > exit_threshold)
    assert pos.iloc[0] == 0.0
    assert pos.iloc[1] == 1.0
    assert pos.iloc[2] == 1.0
    assert pos.iloc[3] == 1.0  # still above exit
    assert pos.iloc[4] == 0.0  # exits
    assert pos.iloc[5] == -1.0  # short entry
    assert pos.iloc[6] == -1.0
    assert pos.iloc[7] == -1.0  # still below -exit
    assert pos.iloc[8] == 0.0


def test_phase_threshold_rejects_bad_thresholds():
    fv = pd.DataFrame({"cos_theta_24h": [0.0]})
    with pytest.raises(ValueError):
        phase_threshold_strategy(fv, enter_threshold=0.1, exit_threshold=0.1)


def test_long_only_baseline_is_all_ones():
    fv = pd.DataFrame({"close": [1, 2, 3]})
    pos = long_only_baseline(fv)
    assert (pos == 1.0).all()
    assert pos.name == "position"


def test_run_backtest_requires_matching_lengths():
    prices = pd.Series([100.0, 101.0])
    positions = pd.Series([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        run_backtest(prices, positions)


def test_proportional_strategy_is_bounded():
    n = 100
    sig = np.sin(np.linspace(0, 4 * np.pi, n))
    conf = np.abs(np.cos(np.linspace(0, 2 * np.pi, n)))
    fv = pd.DataFrame({"cos_theta_24h": sig, "ring_radius": conf})
    pos = proportional_strategy(fv)
    assert (pos >= -1.0).all() and (pos <= 1.0).all()
    assert not pos.isna().any()
    # Position should not be uniformly zero or uniformly one
    assert pos.abs().sum() > 0
    assert pos.abs().mean() < 1.0  # continuous sizing, not always max


def test_proportional_strategy_no_confidence_col():
    fv = pd.DataFrame({"cos_theta_24h": [0.5, -0.3, 0.8, -0.9]})
    pos = proportional_strategy(fv, confidence_col="nonexistent")
    # Falls back to unit confidence — position = clip(signal, -1, 1)
    expected = np.clip([0.5, -0.3, 0.8, -0.9], -1, 1)
    np.testing.assert_allclose(pos.values, expected)


def test_proportional_strategy_reduces_position_when_radius_low():
    fv = pd.DataFrame({
        "cos_theta_24h": [1.0, 1.0],
        "ring_radius":   [1.0, 0.0],
    })
    pos = proportional_strategy(fv)
    # High-confidence bar has larger position than low-confidence bar
    assert pos.iloc[0] > pos.iloc[1]


def test_proportional_deadband_reduces_trades_vs_full():
    n = 200
    sig = np.sin(np.linspace(0, 8 * np.pi, n))
    fv = pd.DataFrame({"cos_theta_24h": sig, "ring_radius": np.ones(n)})
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n))))

    r_full = run_backtest(prices, proportional_strategy(fv))
    r_db = run_backtest(prices, proportional_deadband_strategy(fv, min_trade_size=0.2))
    assert r_db.metrics["n_trades"] < r_full.metrics["n_trades"]


def test_proportional_deadband_holds_position():
    fv = pd.DataFrame({"cos_theta_24h": [0.0, 0.1, 0.12, 0.5, 0.55, 0.3, 0.0]})
    pos = proportional_deadband_strategy(fv, confidence_col="nonexistent", min_trade_size=0.2)
    # First bar: desired=0, held=0
    assert pos.iloc[0] == pytest.approx(0.0)
    # Bars 1,2: desired 0.1/0.12 — delta from 0 is < 0.2, hold 0
    assert pos.iloc[1] == pytest.approx(0.0)
    assert pos.iloc[2] == pytest.approx(0.0)
    # Bar 3: desired=0.5, delta=0.5 >= 0.2 → update to 0.5
    assert pos.iloc[3] == pytest.approx(0.5)
    # Bar 4: desired=0.55, delta=0.05 < 0.2 → still 0.5
    assert pos.iloc[4] == pytest.approx(0.5)
    # Bar 5: desired=0.3, delta=0.2 >= 0.2 → update to 0.3
    assert pos.iloc[5] == pytest.approx(0.3)


def test_regime_gated_filters_by_percentile():
    # Random gate values uniform on [0, 1]. With gate_percentile=0.7,
    # roughly 30% of bars should be "active" (above rolling 70th percentile).
    rng = np.random.default_rng(0)
    n = 2000
    sig = np.sign(np.sin(np.linspace(0, 30 * np.pi, n))).astype(float)
    gate = rng.uniform(0, 1, n)
    fv = pd.DataFrame({"cos_theta_24h": sig, "ring_radius": gate})
    pos = regime_gated_strategy(
        fv, signal_col="cos_theta_24h", gate_col="ring_radius",
        gate_percentile=0.7, gate_window=200, min_trade_size=0.05,
    )
    # Roughly 20-40% of bars should hold a non-zero position
    active_frac = (pos.iloc[300:] != 0.0).mean()
    assert 0.15 < active_frac < 0.45


def test_regime_gated_position_follows_signal_when_active():
    n = 600
    # Constant positive signal, high gate everywhere
    sig = np.ones(n)
    gate = np.linspace(0.5, 1.0, n)  # monotonically rising
    fv = pd.DataFrame({"cos_theta_24h": sig, "ring_radius": gate})
    pos = regime_gated_strategy(
        fv, signal_col="cos_theta_24h", gate_col="ring_radius",
        gate_percentile=0.5, gate_window=100, min_trade_size=0.05,
    )
    # End of series: should be long
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_regime_gated_rejects_bad_args():
    fv = pd.DataFrame({"cos_theta_24h": [0.0], "ring_radius": [1.0]})
    with pytest.raises(ValueError):
        regime_gated_strategy(fv, gate_percentile=1.5)
    with pytest.raises(KeyError):
        regime_gated_strategy(fv, signal_col="nope")
    with pytest.raises(KeyError):
        regime_gated_strategy(fv, gate_col="nope")


def _make_synthetic_features(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    theta = 2 * np.pi * t / 100.0
    rets = 0.02 * np.cos(theta) + rng.normal(0, 0.005, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "sin_theta_6h":  np.sin(theta * 1.0),
        "cos_theta_6h":  np.cos(theta * 1.0),
        "sin_theta_24h": np.sin(theta * 0.5),
        "cos_theta_24h": np.cos(theta * 0.5),
        "sin_theta_72h": np.sin(theta * 0.2),
        "cos_theta_72h": np.cos(theta * 0.2),
        "ring_radius":   np.ones(n),
        "vol_20":        np.full(n, 0.01),
        "lr_z20":        rng.normal(0, 1, n),
        "close":         close,
    })
    return df


def test_build_target_horizon_alignment():
    fv = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    tgt = build_target(fv, horizon=2)
    # tgt[0] = log(102/100), tgt[1] = log(103/101), last two NaN
    assert tgt.iloc[0] == pytest.approx(np.log(102 / 100))
    assert tgt.iloc[1] == pytest.approx(np.log(103 / 101))
    assert np.isnan(tgt.iloc[-2])
    assert np.isnan(tgt.iloc[-1])


def test_model_strategy_zeros_train_slice_and_predicts_oos():
    fv = _make_synthetic_features(n=600)
    result = model_strategy(
        fv, horizon=10, train_frac=0.5, return_diagnostics=True,
    )
    pos = result.positions
    # Train slice (first 300) must all be zero
    assert (pos.iloc[:300] == 0.0).all()
    # Test slice should have nonzero positions if signal is present
    assert pos.iloc[300:].abs().sum() > 0
    # Bounded
    assert (pos >= -1.0).all() and (pos <= 1.0).all()


def test_model_strategy_rejects_missing_features():
    fv = pd.DataFrame({"close": [100, 101], "sin_theta_6h": [0, 0]})
    with pytest.raises(KeyError):
        model_strategy(fv, horizon=1, train_frac=0.5)


def test_model_strategy_rejects_short_train_slice():
    fv = _make_synthetic_features(n=120)
    with pytest.raises(ValueError):
        # train_frac 0.5 of 120 = 60, minus horizon 12 = 48 < 100 minimum
        model_strategy(fv, horizon=12, train_frac=0.5)


def test_metrics_include_verdict_and_sharpe():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0001, 0.01, size=200).tolist()
    prices = _make_prices(rets)
    positions = pd.Series(np.sign(prices.diff().fillna(0)).shift(1).fillna(0.0).values,
                          index=prices.index)
    r = run_backtest(prices, positions)
    assert "verdict" in r.metrics
    assert "net_sharpe" in r.metrics
    assert isinstance(r.summary(), str)
