import json

import numpy as np
import pandas as pd

from eso.cli import main
from eso.lab import dissect_feature_frame, write_asset_dissect_bundle
from eso.modeling import build_feature_frame


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


def test_dissect_feature_frame_returns_tables():
    fv = build_feature_frame(_make_market_df(), feature_mode="compact")
    dissected = dissect_feature_frame(fv, asset_id="unit", horizons=(3, 12), train_size=100)
    result = dissected["result"]
    assert result["schema_version"] == "eso.asset_dissect.v1"
    assert result["asset_id"] == "unit"
    assert not dissected["correlations"].empty
    assert not dissected["quantiles"].empty
    assert result["top_feature_target_correlations"]


def test_dissect_cycle_mode_has_phase_sectors(tmp_path):
    fv = build_feature_frame(_make_market_df(n=180), feature_mode="cycle", umap_train_size=80, smooth_windows=(6, 24))
    dissected = dissect_feature_frame(fv, asset_id="cycle", horizons=(3,), train_size=40)
    assert not dissected["phase_sectors"].empty


def test_write_asset_dissect_bundle(tmp_path):
    fv = build_feature_frame(_make_market_df(), feature_mode="compact")
    dissected = dissect_feature_frame(fv, asset_id="unit", horizons=(3,), train_size=100)
    artifacts = write_asset_dissect_bundle(dissected, tmp_path / "out")
    assert "asset_dissect_json" in artifacts
    payload = json.loads((tmp_path / "out" / "asset_dissect.json").read_text())
    assert payload["schema_version"] == "eso.asset_dissect.v1"
    assert (tmp_path / "out" / "feature_target_correlations.csv").exists()


def test_cli_dissect_writes_bundle(tmp_path):
    csv_path = tmp_path / "market.csv"
    out_dir = tmp_path / "dissect"
    _make_market_df().to_csv(csv_path, index=False)
    code = main([
        "dissect",
        str(csv_path),
        "--feature-mode", "compact",
        "--horizons", "3", "12",
        "--train-size", "100",
        "--output", str(out_dir),
    ])
    assert code == 0
    assert (out_dir / "asset_dissect.json").exists()
    assert main(["validate-json", str(out_dir / "asset_dissect.json")]) == 0
