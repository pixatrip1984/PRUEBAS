"""Tests for rolling signal-stability analyser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eso.diagnostics.signal_stability import analyse_signal_stability


def _features(n: int, regime_flip: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    sin24 = np.sin(2 * np.pi * t / 24)
    cos24 = np.cos(2 * np.pi * t / 24)

    if regime_flip:
        # First half: returns ~ +sin24; second half: returns ~ -sin24 (flipped)
        sign = np.where(t < n // 2, 1.0, -1.0)
        rets = 0.01 * sign * sin24 + rng.normal(0, 0.003, n)
    else:
        # Stable: returns always ~ +sin24
        rets = 0.01 * sin24 + rng.normal(0, 0.003, n)

    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "sin_theta_24h": sin24,
        "cos_theta_24h": cos24,
        "ring_radius": np.ones(n),
        "vol_20": np.full(n, 0.01),
        "lr_z20": rng.normal(0, 1, n),
        "close": close,
    })


def test_stable_signal_has_few_flips():
    fv = _features(n=2000, regime_flip=False, seed=1)
    rep = analyse_signal_stability(fv, horizon=1, window=200)
    # Stable signal: rolling corr should stay positive for sin_theta_24h
    # → few sign flips
    assert rep.flip_count["sin_theta_24h"] <= 5
    assert rep.mean_corr["sin_theta_24h"] > 0


def test_regime_flip_signal_has_many_flips():
    fv = _features(n=2000, regime_flip=True, seed=2)
    rep_stable = analyse_signal_stability(
        _features(n=2000, regime_flip=False, seed=2),
        horizon=1, window=200,
    )
    rep_flip = analyse_signal_stability(fv, horizon=1, window=200)
    # Flipped regime: should have MORE sign flips than stable
    assert rep_flip.flip_count["sin_theta_24h"] > rep_stable.flip_count["sin_theta_24h"]


def test_report_includes_cross_correlations():
    fv = _features(n=1500, regime_flip=True, seed=3)
    rep = analyse_signal_stability(fv, horizon=1, window=200)
    # Cross-corr keys are "{phase}__{regime}"
    keys = list(rep.cross_corr_with_regime.keys())
    assert any("__vol_20" in k for k in keys)
    assert any("__ring_radius" in k for k in keys)


def test_missing_close_raises():
    fv = pd.DataFrame({"sin_theta_24h": [0.0, 0.1]})
    with pytest.raises(KeyError):
        analyse_signal_stability(fv)
