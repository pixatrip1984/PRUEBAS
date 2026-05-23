"""Tests for the live signal generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eso.lab.live_signal import asset_live_signal, get_live_positions, LiveSignal


def _make_csv(tmp_path, n: int = 1500, seed: int = 0,
              filename: str = "TEST_4h.csv") -> str:
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
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def test_asset_live_signal_returns_position(tmp_path):
    csv = _make_csv(tmp_path)
    sig = asset_live_signal(csv)
    assert isinstance(sig, LiveSignal)
    assert -1.0 <= sig.current_position <= 1.0
    assert sig.last_close > 0
    assert sig.selected_signal in {
        "cos_theta_6h", "cos_theta_24h", "cos_theta_72h", "sin_theta_24h",
    }
    assert sig.selected_gate in {0.30, 0.50, 0.70, 0.85}


def test_asset_live_signal_short_data_returns_none(tmp_path):
    csv = _make_csv(tmp_path, n=300)
    sig = asset_live_signal(csv)
    assert sig is None


def test_get_live_positions_writes_output(tmp_path):
    csvs = [
        _make_csv(tmp_path, n=1500, seed=1, filename="A_4h.csv"),
        _make_csv(tmp_path, n=1500, seed=2, filename="B_4h.csv"),
    ]
    out = tmp_path / "live.json"
    rep = get_live_positions(
        configs=[
            {"csv": csvs[0], "include_funding": False},
            {"csv": csvs[1], "include_funding": False},
        ],
        output_path=str(out),
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "per_asset" in payload
    assert "portfolio_positions" in payload
    assert len(payload["per_asset"]) == 2
    # Vol-scaled positions are bounded
    for pos in payload["portfolio_positions"].values():
        assert -1.0 <= pos <= 1.0
