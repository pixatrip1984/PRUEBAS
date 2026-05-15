import numpy as np
import pandas as pd

from eso.data.relative import build_relative_market_features
from eso.lab.context import append_reference_context
from eso.modeling import build_feature_frame


def _asset(n=120, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.uniform(10, 100, n),
        "vwap": close,
        "taker_buy_volume": rng.uniform(3, 50, n),
        "taker_sell_volume": rng.uniform(3, 50, n),
    })


def test_build_relative_market_features_basic_columns():
    asset = _asset(seed=1)
    reference = _asset(seed=2, start=1000)
    out = build_relative_market_features(asset, reference, windows=(12, 24))
    required = {"asset_log_return", "reference_log_return", "relative_log_return",
                "relative_strength_12", "rolling_corr_ref_12", "rolling_beta_ref_24",
                "price_ratio_z_24"}
    assert required.issubset(out.columns)
    assert len(out) > 0


def test_append_reference_context_adds_relative_columns():
    raw = _asset(seed=1)
    reference = _asset(seed=2, start=1000)
    fv = build_feature_frame(raw, feature_mode="compact")
    out = append_reference_context(fv, raw, reference, windows=(12,))
    assert "relative_log_return" in out.columns
    assert "rolling_corr_ref_12" in out.columns
    assert "close" in out.columns
