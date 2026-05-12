"""Tests for eso.signals.cycle — phase extraction and period analysis."""

import numpy as np
import pandas as pd
import pytest

from eso.signals.cycle import analyse_cycle_period, extract_cycle_phase, ring_quality


def _make_synthetic_ring(n: int = 500, noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Synthetic 2D ring: close ≈ A*sin(2π t/T) + drift, vol ≈ |cos| + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    period = 72  # 72 bars ≈ 3 days at 1h
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    # Inject a sine cycle into close so build_financial_features can see it
    close = close * (1 + 0.02 * np.sin(2 * np.pi * t / period))
    volume = 500 + 300 * np.abs(np.cos(2 * np.pi * t / period)) + rng.normal(0, 50, n)
    buy = volume * (0.5 + 0.2 * np.sin(2 * np.pi * t / period + 0.5))
    sell = volume - buy
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="h").astype(str),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.003,
        "low": close * 0.997,
        "close": close,
        "volume": volume,
        "n_trades": rng.integers(100, 5000, n).astype(float),
        "vwap": close * (1 + rng.normal(0, 0.0005, n)),
        "taker_buy_volume": buy,
        "taker_sell_volume": sell,
        "delta": buy - sell,
    })
    return df


def _make_real_like(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Minimal realistic DataFrame without a strong cycle."""
    rng = np.random.default_rng(seed)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    vol = rng.uniform(100, 800, n)
    buy = vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.002, "low": close * 0.998, "close": close,
        "volume": vol, "n_trades": rng.integers(200, 3000, n).astype(float),
        "vwap": close, "taker_buy_volume": buy,
        "taker_sell_volume": vol - buy, "delta": 2 * buy - vol,
    })


class TestExtractCyclePhase:
    def test_returns_dataframe_with_required_columns(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, seed=0)
        for col in ["theta", "cos_theta", "sin_theta", "umap_x", "umap_y", "radius"]:
            assert col in phase.columns, f"Missing column: {col}"

    def test_output_length_matches_valid_features(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, seed=0)
        assert len(phase) > 100  # allow some NaN-drop warmup
        assert len(phase) <= len(df)

    def test_cos_sin_on_unit_circle(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, unwrap=False, seed=0)
        norms = np.sqrt(phase["cos_theta"] ** 2 + phase["sin_theta"] ** 2)
        assert np.allclose(norms, 1.0, atol=1e-6), "cos²+sin²≠1"

    def test_radius_positive(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, seed=0)
        assert (phase["radius"] > 0).all()

    def test_no_nan_in_output(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, seed=0)
        assert not phase[["theta", "cos_theta", "sin_theta"]].isnull().any().any()

    def test_timestamp_column_preserved(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, timestamp_col="timestamp", seed=0)
        assert "timestamp" in phase.columns

    def test_smooth_window_runs(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, smooth_window=5, seed=0)
        assert len(phase) > 0

    def test_unwrap_false_stays_in_pi_range(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, unwrap=False, seed=0)
        assert (phase["theta"].abs() <= np.pi + 1e-9).all()


class TestAnalyseCyclePeriod:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        theta = np.cumsum(rng.normal(0.1, 0.05, 500))  # synthetic unwrapped phase
        result = analyse_cycle_period(pd.Series(theta), bars_per_day=24)
        for key in ["estimated_period_bars", "estimated_period_days",
                    "top_periods_bars", "top_periods_days", "top_periods_named",
                    "angular_velocity_mean", "fft_freqs", "fft_power"]:
            assert key in result, f"Missing key: {key}"

    def test_known_period_detected(self):
        """A clean sine-driven phase should recover its period within 20%."""
        period = 168  # weekly in 1h bars
        n = period * 10
        t = np.arange(n)
        theta = np.cumsum(2 * np.pi / period * np.ones(n))
        result = analyse_cycle_period(pd.Series(theta), bars_per_day=24)
        naive = result["estimated_period_bars"]
        assert abs(naive - period) / period < 0.2, (
            f"Expected ~{period} bars, got {naive:.1f}"
        )

    def test_dataframe_input_accepted(self):
        df = _make_real_like(200)
        phase = extract_cycle_phase(df, seed=0)
        result = analyse_cycle_period(phase, bars_per_day=24)
        assert np.isfinite(result["estimated_period_bars"])

    def test_period_labelling(self):
        """Weekly period should produce a label containing 'weekly'."""
        period = 168
        theta = np.cumsum(2 * np.pi / period * np.ones(period * 8))
        result = analyse_cycle_period(pd.Series(theta), bars_per_day=24)
        assert "weekly" in result["estimated_period_named"], (
            f"Expected weekly label, got: {result['estimated_period_named']}"
        )


class TestRingQuality:
    def test_ring_score_high_for_actual_ring(self):
        """Points uniformly on a circle should score > 0.7."""
        n = 500
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 1.0 + np.random.default_rng(0).normal(0, 0.02, n)
        phase = pd.DataFrame({
            "umap_x": r * np.cos(t),
            "umap_y": r * np.sin(t),
            "radius": r,
            "theta": t,
        })
        q = ring_quality(phase)
        assert q["is_ring"], f"radius_cv={q['radius_cv']:.3f} should be < 0.3"
        assert q["ring_score"] > 0.7

    def test_ring_score_low_for_cloud(self):
        """Uniformly random 2D cloud should not look like a ring."""
        rng = np.random.default_rng(1)
        x, y = rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500)
        r = np.sqrt(x ** 2 + y ** 2)
        phase = pd.DataFrame({"umap_x": x, "umap_y": y, "radius": r, "theta": np.arctan2(y, x)})
        q = ring_quality(phase)
        assert not q["is_ring"], f"Cloud should not be a ring, got cv={q['radius_cv']:.3f}"
