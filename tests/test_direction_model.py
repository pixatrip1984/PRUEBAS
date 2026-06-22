"""Tests for eso.signals.direction_model."""

import numpy as np
import pandas as pd
import pytest

from eso.signals.direction_model import (
    CYCLE_FEATURES,
    DirectionModel,
    build_direction_dataset,
)


def _make_fv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Minimal feature vector-like DataFrame for testing."""
    rng = np.random.default_rng(seed)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    theta = np.linspace(0, 8 * np.pi, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "sin_theta_6h":  np.sin(theta * 1.0),
        "cos_theta_6h":  np.cos(theta * 1.0),
        "sin_theta_24h": np.sin(theta * 0.5),
        "cos_theta_24h": np.cos(theta * 0.5),
        "sin_theta_72h": np.sin(theta * 0.25),
        "cos_theta_72h": np.cos(theta * 0.25),
        "ring_radius": rng.uniform(0.5, 2.0, n),
        "vol_20":  rng.uniform(0.005, 0.02, n),
        "lr_z20":  rng.normal(0, 1, n),
        "log_return": rng.normal(0, 0.002, n),
        "close": close,
    })
    return df


class TestBuildDirectionDataset:
    def test_returns_two_arrays(self):
        fv = _make_fv(300)
        X, y = build_direction_dataset(fv, horizon=12)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_y_only_pm1(self):
        fv = _make_fv(300)
        _, y = build_direction_dataset(fv, horizon=12)
        assert set(y.unique()).issubset({-1, 1})

    def test_length_drops_horizon(self):
        fv = _make_fv(300)
        X, y = build_direction_dataset(fv, horizon=12)
        assert len(X) <= 300 - 12

    def test_feature_cols_subset(self):
        fv = _make_fv(300)
        X, _ = build_direction_dataset(fv, horizon=12)
        for col in X.columns:
            assert col in CYCLE_FEATURES

    def test_no_nan_in_X(self):
        fv = _make_fv(400)
        X, _ = build_direction_dataset(fv, horizon=12)
        assert not X.isnull().any().any()

    def test_custom_feature_cols(self):
        fv = _make_fv(300)
        cols = ["sin_theta_24h", "cos_theta_24h", "vol_20"]
        X, _ = build_direction_dataset(fv, horizon=12, feature_cols=cols)
        assert list(X.columns) == cols


class TestDirectionModel:
    def test_fit_predict_returns_labels(self):
        fv = _make_fv(400)
        X, y = build_direction_dataset(fv, horizon=12)
        model = DirectionModel(model_type="logistic")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({-1, 1})

    def test_predict_before_fit_raises(self):
        fv = _make_fv(200)
        X, _ = build_direction_dataset(fv, horizon=12)
        model = DirectionModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_evaluate_returns_result(self):
        fv = _make_fv(400)
        X, y = build_direction_dataset(fv, horizon=12)
        model = DirectionModel()
        model.fit(X.iloc[:200], y.iloc[:200])
        result = model.evaluate(X.iloc[200:], y.iloc[200:])
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.baseline_accuracy <= 1.0
        assert result.n_test > 0

    def test_feature_importances_present(self):
        fv = _make_fv(400)
        X, y = build_direction_dataset(fv, horizon=12)
        model = DirectionModel(model_type="logistic")
        model.fit(X, y)
        result = model.evaluate(X, y)
        for col in X.columns:
            assert col in result.feature_importances

    def test_predict_proba_in_0_1(self):
        fv = _make_fv(400)
        X, y = build_direction_dataset(fv, horizon=12)
        model = DirectionModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        assert len(proba) == len(X)

    def test_accuracy_float(self):
        fv = _make_fv(400)
        X, y = build_direction_dataset(fv, horizon=12)
        model = DirectionModel()
        model.fit(X.iloc[:200], y.iloc[:200])
        result = model.evaluate(X.iloc[200:], y.iloc[200:])
        assert isinstance(result.accuracy, float)
        assert isinstance(result.accuracy_vs_baseline, float)
