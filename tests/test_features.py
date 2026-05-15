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
    optional_cols = set(groups["derivatives"]) | set(groups["alt_funding"])
    for name, cols in groups.items():
        for col in cols:
            if name in {"derivatives", "alt_funding"}:
                continue
            if col in optional_cols:
                continue
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


# ── prepare_btc_klines ───────────────────────────────────────────────────────

def _make_klines(n: int = 100) -> "pd.DataFrame":
    """Simulate Binance Klines format (as in data/btc_3m/)."""
    rng = np.random.default_rng(0)
    close = 50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    volume = rng.uniform(10, 200, n)
    taker_buy = volume * rng.uniform(0.4, 0.6, n)
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": volume,
        "quote_asset_volume": close * volume,
        "number_of_trades": rng.integers(100, 2000, n).astype(float),
        "taker_buy_base_asset_volume": taker_buy,
        "taker_buy_quote_asset_volume": close * taker_buy,
    })


def test_prepare_btc_klines_renames_columns():
    from eso.data.features import prepare_btc_klines
    df = _make_klines(100)
    out = prepare_btc_klines(df)
    assert "taker_buy_volume" in out.columns
    assert "taker_sell_volume" in out.columns
    assert "delta" in out.columns
    assert "vwap" in out.columns
    assert "n_trades" in out.columns


def test_prepare_btc_klines_taker_sell_sum_equals_volume():
    from eso.data.features import prepare_btc_klines
    df = _make_klines(100)
    out = prepare_btc_klines(df)
    diff = (out["taker_buy_volume"] + out["taker_sell_volume"] - out["volume"]).abs()
    assert diff.max() < 1e-9, f"taker_buy + taker_sell != volume: max diff={diff.max()}"


def test_klines_then_build_features_works():
    from eso.data.features import prepare_btc_klines
    df = _make_klines(200)
    normed = prepare_btc_klines(df)
    feat = build_financial_features(normed)
    required = {"log_return", "vol_20", "vwap_dev", "volume_imbalance"}
    assert required.issubset(set(feat.columns)), f"Missing: {required - set(feat.columns)}"


def test_build_features_with_funding_and_open_interest():
    df = _make_ohlcv(260)
    rng = np.random.default_rng(123)
    df["funding_rate"] = rng.normal(0, 0.0001, len(df))
    df["open_interest"] = 1_000_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(df))))
    feat = build_financial_features(df)
    required = {"funding_rate", "funding_abs", "funding_z", "funding_abs_pct",
                "open_interest", "oi_log_return", "oi_change_20", "oi_z_20"}
    assert required.issubset(set(feat.columns)), f"Missing: {required - set(feat.columns)}"
    tail = feat.tail(40)
    assert not tail[list(required)].isnull().any().any()


def test_alt_funding_group_available_when_columns_present():
    df = _make_ohlcv(260)
    rng = np.random.default_rng(123)
    df["funding_rate"] = rng.normal(0, 0.0001, len(df))
    df["open_interest"] = 1_000_000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(df))))
    feat = build_financial_features(df)
    group = feature_column_groups()["alt_funding"]
    present = [c for c in group if c in feat.columns]
    assert "funding_z" in present
    assert "oi_z_20" in present
