"""Tests for eso.data.features financial feature engineering."""

import numpy as np
import pandas as pd
import pytest

from eso.data.features import build_financial_features, feature_column_groups


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.uniform(100, 1000, n)
    taker_buy = volume * rng.uniform(0.3, 0.7, n)
    taker_sell = volume - taker_buy
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "n_trades": rng.integers(100, 5000, n).astype(float),
        "vwap": close * (1 + rng.normal(0, 0.001, n)),
        "taker_buy_volume": taker_buy, "taker_sell_volume": taker_sell,
        "delta": taker_buy - taker_sell,
    })


def test_build_returns_expected_columns():
    df = _make_ohlcv(200)
    feat = build_financial_features(df)
    expected = {"log_return", "vol_5", "vol_20", "vwap_dev", "volume_imbalance", "taker_ratio"}
    assert expected.issubset(set(feat.columns)), f"Missing: {expected - set(feat.columns)}"


def test_no_nan_after_warmup():
    df = _make_ohlcv(200)
    feat = build_financial_features(df)
    # After the NaN-drop in build_financial_features only log_return rows without NaN remain
    # columns with rolling windows may still have NaN in the first few rows — check the tail
    tail = feat.tail(150)
    na_counts = tail.isnull().sum()
    all_zero = (na_counts == 0).all()
    assert all_zero, f"NaNs in tail:\n{na_counts[na_counts > 0]}"


def test_log_return_is_stationary_proxy():
    """log_return should have near-zero mean and much smaller std than raw close."""
    df = _make_ohlcv(500)
    feat = build_financial_features(df)
    lr = feat["log_return"].dropna()
    assert abs(lr.mean()) < 0.01, f"log_return mean too large: {lr.mean()}"
    assert lr.std() < 0.05, f"log_return std unexpectedly large: {lr.std()}"


def test_volume_imbalance_bounded():
    df = _make_ohlcv(200)
    feat = build_financial_features(df)
    vi = feat["volume_imbalance"].dropna()
    assert vi.abs().max() <= 1.0 + 1e-6, f"volume_imbalance out of [-1,1]: {vi.abs().max()}"


def test_feature_groups_are_subsets_of_full_columns():
    df = _make_ohlcv(200)
    feat = build_financial_features(df)
    groups = feature_column_groups()
    for name, cols in groups.items():
        for col in cols:
            assert col in feat.columns, f"Group '{name}' references missing column '{col}'"


def test_minimal_csv_with_close_only():
    """build_financial_features should work with just a close column."""
    df = pd.DataFrame({"close": [100.0 + i * 0.5 for i in range(50)]})
    feat = build_financial_features(df)
    assert "log_return" in feat.columns
    assert len(feat) > 0


def test_no_inf_values():
    df = _make_ohlcv(200)
    feat = build_financial_features(df)
    assert not np.isinf(feat.values).any(), "Infinite values found in features"
