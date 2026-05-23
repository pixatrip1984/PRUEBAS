"""Smoke tests for the asset-validation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eso.lab.asset_validation import validate_asset


def _make_csv(tmp_path, n: int = 1200, seed: int = 0) -> str:
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
    path = tmp_path / "SYN_4h.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_asset_smoke(tmp_path):
    csv_path = _make_csv(tmp_path, n=1200)
    out_dir = tmp_path / "out"
    report = validate_asset(csv_path, output_dir=str(out_dir))
    assert report.asset == "SYN_4h"
    assert (out_dir / "validation_SYN_4h.json").exists()
    assert (out_dir / "validation_SYN_4h.md").exists()
    assert "first_half" in report.half_split
    assert len(report.sweep) > 0
    assert isinstance(report.verdict, str)
