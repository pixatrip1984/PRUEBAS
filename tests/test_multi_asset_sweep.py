"""Tests for the multi-asset sweep tooling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eso.lab.multi_asset_sweep import (
    build_alt_feature_vector,
    evaluate_asset,
    AssetResult,
)


def _synthetic_ohlcv(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Cyclic returns + noise
    rets = 0.005 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(10, 0.3, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })


def test_build_alt_feature_vector_has_required_columns():
    df = _synthetic_ohlcv(n=1200)
    fv = build_alt_feature_vector(df, train_frac=0.5)
    required = {"sin_theta_6h", "cos_theta_6h", "sin_theta_24h", "cos_theta_24h",
                "sin_theta_72h", "cos_theta_72h", "ring_radius",
                "vol_20", "lr_z20", "close"}
    assert required.issubset(set(fv.columns)), \
        f"Missing: {required - set(fv.columns)}"
    assert len(fv) > 100


def test_build_alt_feature_vector_rejects_no_features():
    df = pd.DataFrame({"close": [100, 101, 102, 103]})  # only close
    with pytest.raises(ValueError):
        build_alt_feature_vector(df, train_frac=0.5)


def test_evaluate_asset_on_synthetic(tmp_path):
    df = _synthetic_ohlcv(n=1200)
    csv_path = tmp_path / "TEST_4h.csv"
    df.to_csv(csv_path, index=False)
    result = evaluate_asset(str(csv_path), bars_per_year=2190)
    assert result is not None
    assert isinstance(result, AssetResult)
    assert result.asset == "TEST_4h"
    assert result.n_test > 0


def test_evaluate_asset_short_data_returns_none(tmp_path):
    df = _synthetic_ohlcv(n=300)
    csv_path = tmp_path / "SHORT_4h.csv"
    df.to_csv(csv_path, index=False)
    result = evaluate_asset(str(csv_path))
    assert result is None
