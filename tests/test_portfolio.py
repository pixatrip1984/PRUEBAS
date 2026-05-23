"""Tests for portfolio combination, vol-targeting, and bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eso.lab.portfolio import (
    bootstrap_sharpe_ci,
    equal_weight_portfolio,
    vol_targeted_portfolio,
)
from eso.lab.walk_forward import WalkForwardResult


def _make_wf_result(asset: str, n: int = 600, mu: float = 0.0005,
                    sigma: float = 0.01, seed: int = 0) -> WalkForwardResult:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n).tolist()
    ts = pd.date_range("2024-06-01", periods=n, freq="4h", tz="UTC").astype(str).tolist()
    return WalkForwardResult(
        asset=asset, n_bars=n * 2, n_folds=4, folds=[],
        aggregate_net_return=0.0, aggregate_gross_sharpe=0.0,
        aggregate_net_sharpe=0.0, aggregate_trades=10,
        verdict="TEST", net_returns=rets, timestamps=ts,
    )


def test_bootstrap_sharpe_ci_returns_three_values():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, 1000)
    point, lo, hi = bootstrap_sharpe_ci(rets, bars_per_year=2190, n_boot=200)
    assert lo <= point <= hi
    assert np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)


def test_bootstrap_sharpe_ci_handles_short_series():
    rets = np.array([0.01, -0.01, 0.005])
    point, lo, hi = bootstrap_sharpe_ci(rets, n_boot=10)
    assert np.isnan(point)


def test_equal_weight_portfolio_combines_two_assets():
    a = _make_wf_result("AAA", mu=0.001, seed=1)
    b = _make_wf_result("BBB", mu=0.0005, seed=2)
    rep = equal_weight_portfolio([a, b], bars_per_year=2190)
    assert set(rep.assets) >= {"AAA", "BBB"}
    assert rep.n_bars_union > 0
    assert rep.avg_assets_active > 0


def test_vol_targeted_scales_high_vol_asset_down(tmp_path):
    # Asset A with high vol; B with low vol
    a = _make_wf_result("HIGH", sigma=0.03, seed=3)
    b = _make_wf_result("LOW", sigma=0.005, seed=4)
    rep = vol_targeted_portfolio(
        [a, b], target_annual_vol=0.30,
        bars_per_year=2190, output_dir=str(tmp_path),
    )
    import json
    payload = json.loads((tmp_path / "portfolio_report.json").read_text())
    scalers = payload["per_asset_scaler"]
    # HIGH should be scaled down (scaler < 1), LOW scaled up (scaler > 1)
    assert scalers["HIGH"] < scalers["LOW"]
