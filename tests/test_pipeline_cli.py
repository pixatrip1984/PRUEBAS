import json
import numpy as np
import pandas as pd

from eso.cli import main
from eso.pipeline import ESOExplorer


def test_pipeline_explore_returns_best():
    t = np.linspace(0, 4 * np.pi, 96)
    data = np.stack([np.cos(t), np.sin(t)], axis=1)
    explorer = ESOExplorer(registry_path="/tmp/eso_test_registry.csv")
    report = explorer.explore(data, manifolds=["circle", "sphere2"], k=4, n_masks=2, save=False)
    assert report["best"] is not None
    assert len(report["evaluations"]) == 2


def test_cli_explore_writes_report(tmp_path):
    csv_path = tmp_path / "harmonic.csv"
    out_dir = tmp_path / "report"
    t = np.linspace(0, 4 * np.pi, 96)
    pd.DataFrame({"x": np.cos(t), "y": np.sin(t)}).to_csv(csv_path, index=False)
    code = main([
        "explore",
        str(csv_path),
        "--columns", "x", "y",
        "--manifolds", "circle", "sphere2",
        "--k", "4",
        "--n-masks", "2",
        "--output", str(out_dir),
        "--no-registry",
    ])
    assert code == 0
    assert (out_dir / "report.html").exists()
    assert (out_dir / "report.json").exists()
    payload = json.loads((out_dir / "report.json").read_text())
    assert payload["schema_version"] == "eso.report.v1"
    assert payload["best"]["manifold"] in {"circle", "sphere2"}
