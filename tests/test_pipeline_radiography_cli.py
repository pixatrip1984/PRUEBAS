from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from eso.cli import main
from eso.pipeline import ESOExplorer


def _ohlcv(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    t = np.arange(n, dtype=float)
    ret = 0.002 * np.sin(2.0 * np.pi * t / 17.0) + rng.normal(0.0, 0.0008, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = 0.002 + 0.0005 * np.sin(2.0 * np.pi * t / 11.0)
    volume = 900.0 + 180.0 * np.sin(2.0 * np.pi * t / 13.0) + rng.normal(0.0, 18.0, n)
    volume = np.maximum(volume, 10.0)
    trades = 70.0 + 12.0 * np.sin(2.0 * np.pi * t / 9.0)
    return pd.DataFrame({
        "open": close * np.exp(-ret),
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": volume,
        "n_trades": trades,
        "vwap": close * (1.0 + rng.normal(0.0, 0.0004, n)),
    })


def test_explorer_backward_compatible_explore(tmp_path):
    t = np.linspace(0.0, 4.0 * np.pi, 80)
    data = np.stack([np.cos(t), np.sin(t)], axis=1)
    explorer = ESOExplorer(registry_path=str(tmp_path / "legacy.csv"))
    report = explorer.explore(
        data,
        manifolds=["circle", "sphere2"],
        k=4,
        n_masks=1,
        seed=7,
        save=False,
    )

    assert report["best"] is not None
    assert len(report["evaluations"]) == 2
    assert set(report) >= {
        "dataset_id",
        "diagnosis",
        "suggested_manifolds",
        "evaluations",
        "best",
        "config",
    }


def test_explore_and_radiograph_persists_relational_lab(tmp_path):
    raw = _ohlcv()
    data = raw[["close", "volume"]].to_numpy(dtype=float)
    db_path = tmp_path / "lab.sqlite3"
    explorer = ESOExplorer(
        registry_path=str(tmp_path / "legacy.csv"),
        use_soft_selector=True,
        lab_db_path=str(db_path),
    )

    report = explorer.explore_and_radiograph(
        data,
        dataset_id="synthetic-4h",
        raw_df=raw,
        scale_s="4h",
        manifolds=["circle", "sphere2"],
        k=4,
        n_masks=1,
        seed=11,
        save=False,
    )

    radiography = report["radiography"]
    assert radiography["dataset_id"] == "synthetic-4h"
    assert radiography["scale_s"] == "4h"
    assert radiography["formal_statements_es"]
    assert "Geometria" in radiography["formal_statements_es"][0]
    assert report["selector_scores"]
    assert report["cascade_and_fingerprint_summary"] is not None
    assert report["crossovers"]

    with sqlite3.connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "datasets_configs",
                "experiments",
                "method_crossovers",
                "structural_radiographies",
            )
        }
    assert all(count > 0 for count in counts.values())


def test_radiograph_cli_writes_formal_json_and_markdown(tmp_path, capsys):
    raw = _ohlcv()
    csv_path = tmp_path / "ohlcv.csv"
    raw.to_csv(csv_path, index=False)

    first_out = tmp_path / "plain"
    code = main([
        "radiograph",
        str(csv_path),
        "--columns", "close", "volume",
        "--manifolds", "circle",
        "--k", "4",
        "--n-masks", "1",
        "--output", str(first_out),
        "--no-registry",
    ])
    assert code == 0
    plain_json = json.loads((first_out / "radiography.json").read_text(encoding="utf-8"))
    plain_md = (first_out / "radiography.md").read_text(encoding="utf-8")
    assert plain_json["formal_statements_es"]
    assert "# Radiografia Matematica:" in plain_md

    exog = pd.DataFrame({"macro_flow": np.roll(raw["volume"].to_numpy(), -2)})
    exog_path = tmp_path / "exogenous.csv"
    exog.to_csv(exog_path, index=False)
    second_out = tmp_path / "coupled"
    lab_db = tmp_path / "cli-lab.sqlite3"

    code = main([
        "radiograph",
        str(csv_path),
        "--columns", "close", "volume",
        "--manifolds", "circle", "sphere2",
        "--k", "4",
        "--n-masks", "1",
        "--scale", "4h",
        "--exogenous", str(exog_path),
        "--use-soft-selector",
        "--lab-db", str(lab_db),
        "--output", str(second_out),
        "--no-registry",
    ])
    assert code == 0

    coupled_json = json.loads((second_out / "radiography.json").read_text(encoding="utf-8"))
    coupled_md = (second_out / "radiography.md").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert coupled_json["formal_statements_es"]
    assert "Los datos a una escala" in captured.out
    assert "Radiografia Matematica" in coupled_md
    assert lab_db.exists()
