import json

import numpy as np
import pandas as pd

from eso.cli import main
from eso.lab import run_universe_dissect, write_universe_dissect_result


def _make_market_df(n=180, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.uniform(10, 50, n)
    buy = volume * rng.uniform(0.35, 0.65, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "vwap": close,
        "taker_buy_volume": buy,
        "taker_sell_volume": volume - buy,
    })


def _universe(path_a, path_b):
    return {
        "schema_version": "eso.asset_universe.v1",
        "name": "unit_universe",
        "defaults": {
            "feature_mode": "compact",
            "horizons": [3, 12],
            "train_size": 80,
        },
        "assets": [
            {"asset_id": "A", "path": str(path_a)},
            {"asset_id": "B", "path": str(path_b)},
        ],
    }


def test_run_universe_dissect_writes_aggregates(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_market_df(seed=1).to_csv(a, index=False)
    _make_market_df(seed=2).to_csv(b, index=False)
    result = run_universe_dissect(_universe(a, b), tmp_path / "out")
    assert result["schema_version"] == "eso.universe_dissect.v1"
    assert result["asset_count"] == 2
    assert result["error_count"] == 0
    assert (tmp_path / "out" / "asset_summary.csv").exists()
    summary = pd.read_csv(tmp_path / "out" / "asset_summary.csv")
    assert set(summary["asset_id"]) == {"A", "B"}


def test_write_universe_dissect_result(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_market_df(seed=1).to_csv(a, index=False)
    _make_market_df(seed=2).to_csv(b, index=False)
    result = run_universe_dissect(_universe(a, b), tmp_path / "out")
    path = tmp_path / "out" / "universe_dissect.json"
    write_universe_dissect_result(result, path)
    assert json.loads(path.read_text())["schema_version"] == "eso.universe_dissect.v1"


def test_cli_dissect_many(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    suite = tmp_path / "universe.json"
    _make_market_df(seed=1).to_csv(a, index=False)
    _make_market_df(seed=2).to_csv(b, index=False)
    suite.write_text(json.dumps(_universe(a, b)), encoding="utf-8")
    code = main(["dissect-many", str(suite), "--output", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / "universe_dissect.json").exists()
    assert main(["validate-json", str(tmp_path / "out" / "universe_dissect.json")]) == 0


def test_universe_dissect_with_reference_context(tmp_path):
    a = tmp_path / "a.csv"
    ref = tmp_path / "btc.csv"
    _make_market_df(seed=1).to_csv(a, index=False)
    _make_market_df(seed=2).to_csv(ref, index=False)
    universe = {
        "schema_version": "eso.asset_universe.v1",
        "name": "with_ref",
        "defaults": {
            "feature_mode": "compact",
            "horizons": [3],
            "train_size": 80,
            "reference_path": str(ref),
            "relative_windows": [12],
        },
        "assets": [{"asset_id": "ALT", "path": str(a)}],
    }
    result = run_universe_dissect(universe, tmp_path / "out_ref")
    assert result["error_count"] == 0
    corr = pd.read_csv(tmp_path / "out_ref" / "ALT" / "feature_target_correlations.csv")
    assert "relative_log_return" in set(corr["feature"])
