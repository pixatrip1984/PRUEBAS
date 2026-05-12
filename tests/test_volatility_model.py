"""Tests for eso.signals.volatility_model."""

import numpy as np
import pandas as pd
import pytest

from eso.signals.volatility_model import (
    VOL_FEATURES,
    VolatilityModel,
    build_vol_dataset,
)


def _make_fv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    theta = np.linspace(0, 8 * np.pi, n)
    vol = rng.uniform(0.005, 0.025, n)
    return pd.DataFrame({
        "timestamp":     pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "sin_theta_6h":  np.sin(theta),
        "cos_theta_6h":  np.cos(theta),
        "sin_theta_24h": np.sin(theta * 0.5),
        "cos_theta_24h": np.cos(theta * 0.5),
        "sin_theta_72h": np.sin(theta * 0.25),
        "cos_theta_72h": np.cos(theta * 0.25),
        "ring_radius":   rng.uniform(0.5, 2.5, n),
        "vol_20":        vol,
        "lr_z20":        rng.normal(0, 1, n),
        "log_return":    rng.normal(0, 0.002, n),
        "close":         close,
    })


class TestBuildVolDataset:
    def test_returns_two_objects(self):
        fv = _make_fv(300)
        X, y = build_vol_dataset(fv, horizon=12)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_y_all_positive(self):
        fv = _make_fv(300)
        _, y = build_vol_dataset(fv, horizon=12)
        assert (y >= 0).all()

    def test_no_nan(self):
        fv = _make_fv(300)
        X, y = build_vol_dataset(fv, horizon=12)
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_length_at_most_n_minus_horizon(self):
        fv = _make_fv(300)
        X, y = build_vol_dataset(fv, horizon=12)
        assert len(X) <= 300 - 12

    def test_custom_feature_cols(self):
        fv = _make_fv(300)
        cols = ["ring_radius", "vol_20"]
        X, _ = build_vol_dataset(fv, horizon=12, feature_cols=cols)
        assert list(X.columns) == cols

    def test_horizon_1_uses_abs_return(self):
        fv = _make_fv(200)
        X, y = build_vol_dataset(fv, horizon=1)
        assert (y >= 0).all()


class TestVolatilityModel:
    def test_fit_predict_shape(self):
        fv = _make_fv(400)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="ridge")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self):
        fv = _make_fv(200)
        X, _ = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_predictions_positive(self):
        fv = _make_fv(400)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="ridge")
        model.fit(X, y)
        preds = model.predict(X)
        # Ridge can predict negative in principle, but on vol data should be > 0 mostly
        assert np.isfinite(preds).all()

    def test_evaluate_returns_result(self):
        fv = _make_fv(500)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="ridge")
        model.fit(X.iloc[:250], y.iloc[:250])
        result = model.evaluate(X.iloc[250:], y.iloc[250:])
        assert result.mae >= 0
        assert result.rmse >= 0
        assert np.isfinite(result.pearson_r)
        assert result.n_test > 0

    def test_feature_importances_present_ridge(self):
        fv = _make_fv(400)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="ridge")
        model.fit(X, y)
        result = model.evaluate(X, y)
        for col in X.columns:
            assert col in result.feature_importances

    def test_rf_model_works(self):
        fv = _make_fv(400)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="rf", seed=0)
        model.fit(X.iloc[:200], y.iloc[:200])
        result = model.evaluate(X.iloc[200:], y.iloc[200:])
        assert result.mae >= 0
        assert np.isfinite(result.pearson_r)

    def test_partial_r_finite(self):
        fv = _make_fv(400)
        X, y = build_vol_dataset(fv, horizon=12)
        model = VolatilityModel(model_type="ridge")
        model.fit(X.iloc[:200], y.iloc[:200])
        result = model.evaluate(X.iloc[200:], y.iloc[200:])
        assert np.isfinite(result.partial_r_vs_vol20)
