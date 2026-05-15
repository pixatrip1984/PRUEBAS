import json

import numpy as np
import pandas as pd

from eso.benchmark import run_benchmark_suite, write_benchmark_result
from eso.cli import main


def _make_market_df(n=220, seed=0):
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


def _suite(csv_path):
    return {
        "schema_version": "eso.benchmark_suite.v1",
        "name": "unit_suite",
        "dataset_id": "unit_market",
        "dataset": {"path": str(csv_path), "feature_mode": "compact", "close_col": "close"},
        "train_size": 100,
        "runs": [
            {"name": "vol_h6", "task": "volatility", "horizon": 6, "model_type": "ridge"},
            {"name": "regime_h6", "task": "regime", "horizon": 6, "model_type": "logistic"},
            {"name": "dir_h3", "task": "direction", "horizon": 3, "model_type": "logistic"},
        ],
    }


def test_run_benchmark_suite_writes_run_manifests(tmp_path):
    csv_path = tmp_path / "market.csv"
    _make_market_df().to_csv(csv_path, index=False)
    result = run_benchmark_suite(
        _suite(csv_path),
        output_dir=tmp_path / "bench",
        registry_path=tmp_path / "runs.csv",
    )
    assert result["schema_version"] == "eso.benchmark_result.v1"
    assert result["run_count"] == 3
    assert result["error_count"] == 0
    for run in result["runs"]:
        assert run["manifest_path"]


def test_write_benchmark_result_validates_contract(tmp_path):
    csv_path = tmp_path / "market.csv"
    _make_market_df().to_csv(csv_path, index=False)
    result = run_benchmark_suite(_suite(csv_path), tmp_path / "bench", save_registry=False)
    out = tmp_path / "benchmark_result.json"
    write_benchmark_result(result, out)
    assert json.loads(out.read_text())["schema_version"] == "eso.benchmark_result.v1"


def test_cli_benchmark_runs_suite(tmp_path):
    csv_path = tmp_path / "market.csv"
    suite_path = tmp_path / "suite.json"
    out_dir = tmp_path / "bench"
    _make_market_df().to_csv(csv_path, index=False)
    suite_path.write_text(json.dumps(_suite(csv_path)), encoding="utf-8")
    code = main([
        "benchmark",
        str(suite_path),
        "--output", str(out_dir),
        "--registry", str(tmp_path / "runs.csv"),
    ])
    assert code == 0
    assert (out_dir / "benchmark_result.json").exists()


def test_benchmark_suite_supports_cycle_feature_mode(tmp_path):
    csv_path = tmp_path / "market.csv"
    _make_market_df(n=180).to_csv(csv_path, index=False)
    suite = _suite(csv_path)
    suite["dataset"]["feature_mode"] = "cycle"
    suite["dataset"]["umap_train_size"] = 80
    suite["dataset"]["smooth_windows"] = [6, 12]
    suite["train_size"] = 40
    result = run_benchmark_suite(suite, tmp_path / "bench_cycle", save_registry=False)
    assert result["error_count"] == 0
    assert result["run_count"] == 3
    assert "ring_radius" in result["runs"][0]["features"]
