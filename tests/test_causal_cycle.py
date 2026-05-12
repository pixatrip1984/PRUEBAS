"""Tests for eso.signals.causal_cycle — causal phase extraction and validation."""

import numpy as np
import pandas as pd
import pytest

from eso.signals.causal_cycle import CausalCyclePhase, RollingCausalCycle


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    vol = rng.uniform(200, 800, n)
    buy = vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="h").astype(str),
        "open":  close * (1 + rng.normal(0, 0.001, n)),
        "high":  close * 1.003,
        "low":   close * 0.997,
        "close": close,
        "volume": vol,
        "n_trades": rng.integers(200, 3000, n).astype(float),
        "vwap":  close * (1 + rng.normal(0, 0.0005, n)),
        "taker_buy_volume":  buy,
        "taker_sell_volume": vol - buy,
        "delta": 2 * buy - vol,
    })


# ── CausalCyclePhase ──────────────────────────────────────────────────────────

class TestCausalCyclePhase:
    def test_fit_does_not_raise(self):
        df = _make_df(400)
        from eso.data.features import build_financial_features, feature_column_groups
        feat = build_financial_features(df).dropna()
        cols = [c for c in feature_column_groups()["compact"] if c in feat.columns]
        causal = CausalCyclePhase(train_size=200, seed=0)
        causal.fit(feat[cols].iloc[:200])

    def test_transform_shape_matches_input(self):
        df = _make_df(500)
        from eso.data.features import build_financial_features, feature_column_groups
        feat = build_financial_features(df).dropna()
        cols = [c for c in feature_column_groups()["compact"] if c in feat.columns]
        causal = CausalCyclePhase(train_size=200, seed=0)
        causal.fit(feat[cols].iloc[:200])
        emb = causal.transform(feat[cols].iloc[200:300])
        assert emb.shape == (100, 2)
        assert np.isfinite(emb).all()

    def test_transform_before_fit_raises(self):
        df = _make_df(300)
        from eso.data.features import build_financial_features, feature_column_groups
        feat = build_financial_features(df).dropna()
        cols = [c for c in feature_column_groups()["compact"] if c in feat.columns]
        with pytest.raises(RuntimeError, match="fit"):
            CausalCyclePhase().transform(feat[cols])

    def test_embedding_to_phase_columns(self):
        emb = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 50)),
                                np.sin(np.linspace(0, 2*np.pi, 50))])
        phase = CausalCyclePhase.embedding_to_phase(emb, unwrap=False)
        for col in ["theta", "cos_theta", "sin_theta", "umap_x", "umap_y", "radius"]:
            assert col in phase.columns

    def test_cos_sin_unit_norm(self):
        emb = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 50)),
                                np.sin(np.linspace(0, 2*np.pi, 50))])
        phase = CausalCyclePhase.embedding_to_phase(emb, unwrap=False)
        norms = np.sqrt(phase["cos_theta"]**2 + phase["sin_theta"]**2)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_fit_and_evaluate_returns_result(self):
        df = _make_df(500)
        causal = CausalCyclePhase(train_size=200, seed=0)
        result = causal.fit_and_evaluate(df)
        assert hasattr(result, "test_ring_cv")
        assert hasattr(result, "correlations")
        assert result.test_size > 0
        assert result.train_size == 200
        assert np.isfinite(result.test_ring_cv)

    def test_oos_phase_has_no_leakage(self):
        """Test that OOS phase index is strictly after train_size rows."""
        df = _make_df(500)
        causal = CausalCyclePhase(train_size=200, seed=0)
        result = causal.fit_and_evaluate(df)
        # Test phase should only use feature rows after train_size
        assert result.test_phase.index.min() >= 200

    def test_train_and_test_sizes_sum_correctly(self):
        df = _make_df(500)
        from eso.data.features import build_financial_features
        n_valid = len(build_financial_features(df).dropna())
        causal = CausalCyclePhase(train_size=200, seed=0)
        result = causal.fit_and_evaluate(df)
        assert result.train_size + result.test_size == n_valid


# ── RollingCausalCycle ────────────────────────────────────────────────────────

class TestRollingCausalCycle:
    def test_fit_predict_returns_dataframe(self):
        df = _make_df(600)
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        assert isinstance(phase, pd.DataFrame)
        assert len(phase) > 0

    def test_output_only_after_init_train(self):
        df = _make_df(600)
        from eso.data.features import build_financial_features
        feat = build_financial_features(df).dropna()
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        assert phase.index.min() >= 200

    def test_output_columns_present(self):
        df = _make_df(600)
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        for col in ["theta", "cos_theta", "sin_theta", "radius"]:
            assert col in phase.columns

    def test_no_nan_in_output(self):
        df = _make_df(600)
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        assert not phase[["theta", "cos_theta", "sin_theta"]].isnull().any().any()

    def test_cos_sin_unit_norm(self):
        df = _make_df(600)
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        norms = np.sqrt(phase["cos_theta"]**2 + phase["sin_theta"]**2)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_covers_full_post_init_range(self):
        """All bars after init_train should appear in the output."""
        df = _make_df(600)
        from eso.data.features import build_financial_features
        feat = build_financial_features(df).dropna()
        n = len(feat)
        rolling = RollingCausalCycle(init_train=200, refit_every=100, seed=0)
        phase = rolling.fit_predict(df)
        assert len(phase) == n - 200
