from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from eso.lab.evolution_engine import AutonomousMathLab


def _ohlcv(seed: int = 17, n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    ret = 0.0015 * np.sin(2.0 * np.pi * t / 19.0) + rng.normal(0.0, 0.0008, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = 0.0015 + 0.0004 * np.cos(2.0 * np.pi * t / 13.0)
    volume = 900.0 + 180.0 * np.sin(2.0 * np.pi * t / 11.0) + rng.normal(0.0, 25.0, n)
    volume = np.clip(volume, 100.0, None)
    trades = 70.0 + 12.0 * np.sin(2.0 * np.pi * t / 7.0)
    imbalance = 0.35 * np.sin(2.0 * np.pi * t / 17.0)
    buy = volume * (1.0 + imbalance) / 2.0
    sell = volume - buy
    return pd.DataFrame(
        {
            "open": close * np.exp(-ret),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": volume,
            "n_trades": trades,
            "vwap": close * (1.0 + rng.normal(0.0, 0.00025, n)),
            "taker_buy_volume": buy,
            "taker_sell_volume": sell,
            "delta": buy - sell,
        }
    )


def _peer_and_exogenous(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    t = np.arange(n, dtype=float)
    driver = np.sin(2.0 * np.pi * t / 15.0)
    peer = pd.DataFrame(
        {
            "BTC": driver,
            "ALT": np.roll(driver, 2) + 0.01 * np.cos(2.0 * np.pi * t / 9.0),
        },
        index=frame.index,
    )
    exogenous = pd.DataFrame(
        {
            "macro_flow": np.roll(frame["close"].to_numpy(), -1),
        },
        index=frame.index,
    )
    return peer, exogenous


def _table_count(path, table: str) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_run_cycle_persists_full_evolution_and_radiography(tmp_path):
    db_path = tmp_path / "lab.sqlite3"
    frame = _ohlcv()
    peer, exogenous = _peer_and_exogenous(frame)
    lab = AutonomousMathLab(
        store_path=db_path,
        epsilon=0.01,
        top_k_manifolds=2,
        crossover_limit=2,
        seed=123,
    )

    result = lab.run_cycle(
        frame,
        dataset_id="btc-synthetic",
        scale_s="4h",
        peer_series_df=peer,
        exogenous_df=exogenous,
        k=4,
        n_masks=1,
        projection_method="linear",
    )
    json.dumps(result, allow_nan=False)

    assert _table_count(db_path, "datasets_configs") >= 1
    assert _table_count(db_path, "experiments") >= 2
    assert _table_count(db_path, "method_crossovers") >= 1
    assert _table_count(db_path, "structural_radiographies") >= 1

    assert min(result["selection_weights"].values()) >= lab.epsilon
    assert len(result["selection_weights"]) >= 23
    assert all(
        item["method_id"]
        not in {
            "spectral_causal",
            "persistent_homology",
            "microstructure_flow",
            "cascade_branching",
        }
        for item in result["evaluations"]
    )

    crossovers = lab.store.query_crossovers()
    assert crossovers
    assert all(item["status"] in {"validated", "rejected"} for item in crossovers)
    assert all("hybrid_weighted_score" in item["metrics"] for item in crossovers)

    assert set(result["fingerprint_summary"]) == {
        "stealth_accumulation_index",
        "vpin",
        "kyle_lambda",
        "order_splitting_fingerprint",
    }
    assert all(
        {"mean", "max"} <= set(summary)
        for summary in result["fingerprint_summary"].values()
    )
    assert "branching_ratio_eta" in result["cascade_summary"]
    assert "avalanche_count" in result["cascade_summary"]
    assert result["capital_profiles"]["regime_counts"]
    assert result["ecosystem_couplings"]

    radiography = result["radiography"]
    assert radiography["dataset_id"] == "btc-synthetic"
    assert radiography["formal_statements_es"]
    assert radiography["formal_statements_en"]
    assert radiography["scientific_statements"]["es"]
    assert lab.store.query_radiographies(dataset_id="btc-synthetic")


def test_evolve_multiple_reinforces_historical_selector_scores(tmp_path):
    rng = np.random.default_rng(44)
    first = rng.normal(size=(72, 3))
    second = first + rng.normal(scale=0.08, size=first.shape)
    lab = AutonomousMathLab(
        store_path=tmp_path / "history.sqlite3",
        epsilon=0.01,
        top_k_manifolds=1,
        crossover_limit=1,
        seed=7,
    )

    summary = lab.evolve_multiple(
        [
            {"df_or_array": first, "dataset_id": "first", "scale_s": "window-72"},
            {"df_or_array": second, "dataset_id": "second", "scale_s": "window-72"},
        ],
        k=3,
        n_masks=1,
        projection_method="linear",
    )
    json.dumps(summary, allow_nan=False)

    assert summary["cycle_count"] == 2
    first_history = max(
        row["historical_score"]
        for row in summary["cycles"][0]["selection_ranking"]
    )
    second_history = max(
        row["historical_score"]
        for row in summary["cycles"][1]["selection_ranking"]
    )
    assert first_history == 0.0
    assert second_history > first_history
    assert summary["family_performance"]
    assert any(
        row["experiment_count"] >= 1
        for row in summary["family_performance"]
    )


def test_non_ohlcv_ndarray_runs_without_microstructure_columns(tmp_path):
    rng = np.random.default_rng(91)
    data = rng.normal(size=(64, 2))
    lab = AutonomousMathLab(
        store_path=tmp_path / "array.sqlite3",
        top_k_manifolds=1,
        crossover_limit=1,
        seed=11,
    )

    result = lab.run_cycle(
        data,
        dataset_id="matrix-only",
        k=3,
        n_masks=1,
        projection_method="linear",
    )
    json.dumps(result, allow_nan=False)

    assert result["evaluations"]
    assert result["fingerprint_summary"] == {}
    assert result["cascade_summary"] == {}
    assert result["capital_profiles"] == {}
    assert result["radiography"]["formal_statements_es"]
