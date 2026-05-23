"""Smoke tests for walk-forward backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eso.lab.walk_forward import walk_forward_backtest


def _make_csv(tmp_path, n: int = 1500, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    rets = 0.005 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(10, 0.3, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    path = tmp_path / "WF_4h.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_walk_forward_smoke(tmp_path):
    csv = _make_csv(tmp_path, n=1500)
    result = walk_forward_backtest(csv, n_folds=4, output_dir=str(tmp_path / "out"))
    assert result.asset == "WF_4h"
    # 3 trading folds (fold 0 is warmup)
    assert len(result.folds) == 3
    assert (tmp_path / "out" / "walkforward_WF_4h.json").exists()
    assert (tmp_path / "out" / "walkforward_WF_4h.md").exists()
    assert isinstance(result.verdict, str)


def test_walk_forward_rejects_short_data(tmp_path):
    csv = _make_csv(tmp_path, n=400)
    with pytest.raises(ValueError):
        walk_forward_backtest(csv, n_folds=4)
