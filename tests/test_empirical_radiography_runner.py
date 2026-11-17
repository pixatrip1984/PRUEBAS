from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_math_lab_radiography import main, run_empirical_math_lab


def _write_synthetic_market(data_dir: Path, n: int = 260) -> None:
    rng = np.random.default_rng(2026)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")

    btc_ret = rng.normal(0.0, 0.003, n)
    btc_ret[80:120] += 0.002 * np.sin(np.arange(40) / 3.0)
    btc_close = 50000.0 * np.exp(np.cumsum(btc_ret))
    btc_volume = np.exp(8.0 + 18.0 * np.abs(btc_ret) + rng.normal(0.0, 0.12, n))
    imbalance = np.tanh(80.0 * btc_ret + rng.normal(0.0, 0.12, n))
    buy = btc_volume * (1.0 + imbalance) / 2.0
    sell = btc_volume - buy

    btc = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": btc_close * np.exp(-btc_ret),
            "high": btc_close * (1.0 + 0.0025),
            "low": btc_close * (1.0 - 0.0025),
            "close": btc_close,
            "volume": btc_volume,
            "n_trades": rng.integers(80, 220, n),
            "vwap": btc_close * (1.0 + rng.normal(0.0, 0.0004, n)),
            "volume_usd": btc_volume * btc_close,
            "taker_buy_volume": buy,
            "taker_sell_volume": sell,
            "delta": buy - sell,
        }
    )
    btc.to_csv(data_dir / "BTCUSDT_4h_full.csv", index=False)

    alt1_ret = rng.normal(0.0, 0.0002, n)
    alt1_ret[2:] += 0.95 * btc_ret[:-2]
    alt1_close = 120.0 * np.exp(np.cumsum(alt1_ret))
    funding1 = np.empty(n)
    funding1[:-2] = btc_ret[2:]
    funding1[-2:] = rng.normal(0.0, 0.003, 2)
    gno = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": alt1_close * np.exp(-alt1_ret),
            "high": alt1_close * 1.004,
            "low": alt1_close * 0.996,
            "close": alt1_close,
            "volume": np.exp(6.5 + rng.normal(0.0, 0.2, n)),
            "funding_rate": funding1,
        }
    )
    gno.to_csv(data_dir / "GNOUSDT_4h.csv", index=False)

    alt2_ret = rng.normal(0.0, 0.00025, n)
    alt2_ret[3:] -= 0.9 * btc_ret[:-3]
    alt2_close = 2.0 * np.exp(np.cumsum(alt2_ret))
    funding2 = rng.normal(0.0, 0.002, n)
    funding2[60:180] = btc_ret[61:181] + rng.normal(0.0, 0.0001, 120)
    grass = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "open": alt2_close * np.exp(-alt2_ret),
            "high": alt2_close * 1.005,
            "low": alt2_close * 0.995,
            "close": alt2_close,
            "volume": np.exp(6.2 + rng.normal(0.0, 0.22, n)),
            "funding_rate": funding2,
        }
    )
    grass.to_csv(data_dir / "GRASSUSDT_4h.csv", index=False)


def test_run_empirical_math_lab_synthetic_persists_structural_radiography(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    db_path = tmp_path / "experiments" / "lab.sqlite3"
    data_dir.mkdir()
    _write_synthetic_market(data_dir)

    result = run_empirical_math_lab(
        data_dir=data_dir,
        output_dir=output_dir,
        db_path=db_path,
        max_rows=220,
        k=2,
        n_masks=1,
        seed=17,
        epsilon=0.01,
    )

    assert Path(result["artifacts"]["json"]).is_file()
    assert Path(result["artifacts"]["markdown"]).is_file()
    payload = json.loads(Path(result["artifacts"]["json"]).read_text(encoding="utf-8"))
    assert payload["dataset"]["primary_file"] == "BTCUSDT_4h_full.csv"
    assert len(payload["dataset"]["altcoin_files"]) == 2
    assert min(item["probability"] for item in payload["selector_probabilities"]) >= 0.01
    assert payload["evaluated_manifolds"]
    assert payload["crossovers"]
    assert payload["ecosystem_graph"]["edges"]
    assert payload["exogenous_regimes"]
    assert any(
        item["regime"] != "breakdown" for item in payload["exogenous_regimes"]
    )
    assert payload["cascade"]["branching_ratio_eta"] >= 0.0
    assert "order_splitting_fingerprint" in payload["microstructure_fingerprints"]
    assert payload["radiography"]["scientific_statements"]["es"]

    with sqlite3.connect(db_path) as conn:
        for table in (
            "datasets_configs",
            "experiments",
            "method_crossovers",
            "structural_radiographies",
        ):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert count > 0, table
        statuses = {
            row[0]
            for row in conn.execute("SELECT DISTINCT status FROM method_crossovers").fetchall()
        }
        assert "evaluated" in statuses


def test_main_runs_cleanly_with_bounded_empirical_dataset(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "cli-report"
    db_path = tmp_path / "cli.sqlite3"
    data_dir.mkdir()
    _write_synthetic_market(data_dir)

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--db-path",
            str(db_path),
            "--max-rows",
            "240",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "empirical_radiography.json").is_file()
    assert (output_dir / "empirical_radiography.md").is_file()
    assert db_path.is_file()


def test_repo_data_slices_when_available(tmp_path):
    source = Path("data")
    primary = source / "BTCUSDT_4h_full.csv"
    if not primary.is_file():
        pytest.skip("repository empirical data is not present")

    data_dir = tmp_path / "real-slices"
    data_dir.mkdir()
    primary_frame = pd.read_csv(primary).tail(180)
    primary_frame.to_csv(data_dir / primary.name, index=False)

    copied_alt = 0
    for path in sorted(source.glob("*_4h.csv"))[:2]:
        pd.read_csv(path).tail(180).to_csv(data_dir / path.name, index=False)
        copied_alt += 1
    if copied_alt == 0:
        pytest.skip("repository altcoin data is not present")

    result = run_empirical_math_lab(
        data_dir=data_dir,
        output_dir=tmp_path / "real-report",
        db_path=tmp_path / "real.sqlite3",
        max_rows=160,
        k=1,
        n_masks=1,
        seed=29,
    )

    assert result["evaluated_manifolds"]
    assert result["radiography"]["scientific_statements"]["es"]
    assert Path(result["artifacts"]["json"]).is_file()
