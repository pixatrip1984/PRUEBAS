"""Tests for eso.signals.feature_vector — final ML feature composition."""

import numpy as np
import pandas as pd
import pytest

from eso.signals.feature_vector import (
    FEATURE_COLUMNS,
    CausalFeatureBuilder,
    build_feature_vector,
)


def _make_df(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 35000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    vol = rng.uniform(100, 600, n)
    buy = vol * rng.uniform(0.35, 0.65, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="h").astype(str),
        "open":  close * (1 + rng.normal(0, 0.001, n)),
        "high":  close * 1.002,
        "low":   close * 0.998,
        "close": close,
        "volume": vol,
        "n_trades": rng.integers(300, 4000, n).astype(float),
        "vwap":  close,
        "taker_buy_volume":  buy,
        "taker_sell_volume": vol - buy,
        "delta": 2 * buy - vol,
    })


class TestBuildFeatureVector:
    def test_returns_dataframe(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6, 24))
        assert isinstance(fv, pd.DataFrame)
        assert len(fv) > 0

    def test_core_columns_present(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6, 24))
        for col in ["sin_theta_6h", "cos_theta_6h", "sin_theta_24h", "cos_theta_24h",
                    "ring_radius", "vol_20", "lr_z20", "close", "log_return"]:
            assert col in fv.columns, f"Missing: {col}"

    def test_sin_cos_on_unit_circle(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6, 24))
        for win in [6, 24]:
            norms = np.sqrt(fv[f"sin_theta_{win}h"]**2 + fv[f"cos_theta_{win}h"]**2)
            assert np.allclose(norms, 1.0, atol=1e-5), f"Unit circle violated for {win}h"

    def test_ring_radius_positive(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6, 24))
        assert (fv["ring_radius"] > 0).all()

    def test_oos_only_no_leakage(self):
        """The feature vector should only contain OOS rows (after train_size)."""
        df = _make_df(600)
        from eso.data.features import build_financial_features, feature_column_groups
        feat = build_financial_features(df)
        cols = [c for c in feature_column_groups()["compact"] if c in feat.columns]
        n_valid = len(feat.dropna(subset=cols))
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6,), drop_warmup=False)
        assert len(fv) <= n_valid - 300

    def test_no_inf_values(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6, 24))
        numeric = fv.select_dtypes(include=float)
        assert not np.isinf(numeric.values).any()

    def test_timestamp_column_present(self):
        df = _make_df(600)
        fv = build_feature_vector(df, train_size=300, smooth_windows=(6,))
        assert "timestamp" in fv.columns

    def test_drop_warmup_reduces_rows(self):
        df = _make_df(600)
        fv_full = build_feature_vector(df, train_size=300, smooth_windows=(24,), drop_warmup=False)
        fv_trim = build_feature_vector(df, train_size=300, smooth_windows=(24,), drop_warmup=True)
        assert len(fv_trim) < len(fv_full)


class TestCausalFeatureBuilder:
    def test_returns_dataframe(self):
        df = _make_df(700)
        builder = CausalFeatureBuilder(init_train=200, refit_every=100,
                                        smooth_windows=(6, 24), seed=0)
        fv = builder.fit_predict(df)
        assert isinstance(fv, pd.DataFrame)
        assert len(fv) > 0

    def test_feature_columns_property(self):
        builder = CausalFeatureBuilder(smooth_windows=(6, 24, 72))
        cols = builder.feature_columns
        assert "sin_theta_6h"  in cols
        assert "cos_theta_24h" in cols
        assert "ring_radius"   in cols
        assert "vol_20"        in cols

    def test_sin_cos_unit_circle(self):
        df = _make_df(700)
        builder = CausalFeatureBuilder(init_train=200, refit_every=100,
                                        smooth_windows=(6, 24), seed=0)
        fv = builder.fit_predict(df)
        for win in [6, 24]:
            norms = np.sqrt(fv[f"sin_theta_{win}h"]**2 + fv[f"cos_theta_{win}h"]**2)
            assert np.allclose(norms, 1.0, atol=1e-5), f"Unit circle violated {win}h"

    def test_no_nan_in_core_columns(self):
        df = _make_df(700)
        builder = CausalFeatureBuilder(init_train=200, refit_every=100,
                                        smooth_windows=(6, 24), seed=0)
        fv = builder.fit_predict(df)
        core = [c for c in builder.feature_columns if c in fv.columns]
        assert not fv[core].isnull().any().any()

    def test_output_strictly_after_init_train(self):
        df = _make_df(700)
        from eso.data.features import build_financial_features
        feat = build_financial_features(df).dropna()
        builder = CausalFeatureBuilder(init_train=200, refit_every=100,
                                        smooth_windows=(6,), seed=0)
        fv = builder.fit_predict(df, drop_warmup=False)
        assert fv.index.min() >= 200
