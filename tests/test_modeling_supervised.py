import json

import numpy as np
import pandas as pd

from eso.cli import main
from eso.modeling import build_feature_frame, evaluate_supervised_task


def _make_market_df(n=240, seed=0):
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


def test_build_feature_frame_compact_preserves_close_alignment():
    raw = _make_market_df()
    fv = build_feature_frame(raw, feature_mode="compact")
    assert {"log_return", "vol_20", "vwap_dev", "volume_imbalance", "close"}.issubset(fv.columns)
    assert np.allclose(fv["close"].values, raw.loc[fv.index, "close"].values)


def test_evaluate_supervised_volatility_manifest():
    fv = build_feature_frame(_make_market_df(), feature_mode="compact")
    manifest = evaluate_supervised_task(
        fv,
        task="volatility",
        horizon=6,
        train_size=120,
        dataset_id="unit",
    )
    assert manifest["schema_version"] == "eso.model_run.v1"
    assert manifest["task"] == "volatility"
    assert manifest["split"]["shuffle"] is False
    assert "vol_20_persistence_mae" in manifest["baselines"]
    assert len(manifest["feature_frame"]["sha256"]) == 64


def test_evaluate_supervised_regime_manifest():
    fv = build_feature_frame(_make_market_df(), feature_mode="compact")
    manifest = evaluate_supervised_task(fv, task="regime", horizon=6, train_size=120)
    assert manifest["target"]["type"] == "classification"
    assert 0 <= manifest["metrics"]["accuracy"] <= 1
    assert manifest["model_type"] == "logistic"


def test_evaluate_supervised_direction_manifest():
    fv = build_feature_frame(_make_market_df(), feature_mode="compact")
    manifest = evaluate_supervised_task(fv, task="direction", horizon=3, train_size=120)
    assert manifest["target"]["name"] == "future_direction_3"
    assert "accuracy_vs_baseline" in manifest["metrics"]


def test_build_feature_frame_cycle_mode():
    raw = _make_market_df(n=180)
    fv = build_feature_frame(raw, feature_mode="cycle", umap_train_size=80, smooth_windows=(6, 12))
    assert {"sin_theta_6h", "cos_theta_12h", "ring_radius", "vol_20", "close"}.issubset(fv.columns)
    assert len(fv) > 0


def test_cli_model_eval_writes_manifest(tmp_path):
    csv_path = tmp_path / "market.csv"
    out_dir = tmp_path / "model"
    registry = tmp_path / "runs.csv"
    _make_market_df().to_csv(csv_path, index=False)
    code = main([
        "model-eval",
        str(csv_path),
        "--task", "volatility",
        "--feature-mode", "compact",
        "--horizon", "6",
        "--train-size", "120",
        "--output", str(out_dir),
        "--registry", str(registry),
    ])
    assert code == 0
    path = out_dir / "model_run_manifest.json"
    assert path.exists()
    assert registry.exists()
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == "eso.model_run.v1"
    assert manifest["target"]["name"] == "future_rv_6"


def test_cli_model_runs_list_and_compare(tmp_path):
    csv_path = tmp_path / "market.csv"
    registry = tmp_path / "runs.csv"
    _make_market_df().to_csv(csv_path, index=False)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main([
        "model-eval", str(csv_path), "--task", "direction", "--feature-mode", "compact",
        "--horizon", "3", "--train-size", "120", "--output", str(out_a), "--registry", str(registry),
    ]) == 0
    assert main([
        "model-eval", str(csv_path), "--task", "direction", "--feature-mode", "compact",
        "--horizon", "6", "--train-size", "120", "--output", str(out_b), "--registry", str(registry),
    ]) == 0
    assert main(["model-runs", "list", "--registry", str(registry)]) == 0
    assert main([
        "model-runs", "compare",
        str(out_a / "model_run_manifest.json"),
        str(out_b / "model_run_manifest.json"),
        "--registry", str(registry),
    ]) == 0


def test_cli_validate_json_accepts_model_manifest(tmp_path):
    csv_path = tmp_path / "market.csv"
    out_dir = tmp_path / "model"
    _make_market_df().to_csv(csv_path, index=False)
    assert main([
        "model-eval", str(csv_path), "--task", "volatility", "--feature-mode", "compact",
        "--horizon", "6", "--train-size", "120", "--output", str(out_dir), "--no-registry",
    ]) == 0
    assert main(["validate-json", str(out_dir / "model_run_manifest.json")]) == 0
