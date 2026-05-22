import argparse
import json

from eso.cli import cmd_model_runs
from eso.modeling.registry import compare_model_runs, list_model_runs, rank_model_runs, save_model_run


def _manifest(value=0.1, metric="accuracy_vs_baseline", task="direction"):
    return {
        "schema_version": "eso.model_run.v1",
        "dataset_id": "unit",
        "task": task,
        "model_type": "logistic",
        "features": ["x", "y"],
        "feature_frame": {"sha256": "a" * 64},
        "target": {"name": "future_direction_3", "type": "classification", "horizon": 3},
        "split": {"train_size": 10, "test_size": 5, "shuffle": False},
        "baselines": {"majority_class_accuracy": 0.5},
        "metrics": {metric: value, "accuracy": 0.6},
        "primary_metric": metric,
    }


def test_save_and_list_model_run(tmp_path):
    registry = tmp_path / "runs.csv"
    row = save_model_run(_manifest(), registry, tmp_path / "a.json")
    rows = list_model_runs(registry)
    assert len(rows) == 1
    assert rows[0]["run_id"] == row["run_id"]
    assert rows[0]["primary_metric"] == "accuracy_vs_baseline"


def test_save_model_run_replaces_duplicate(tmp_path):
    registry = tmp_path / "runs.csv"
    manifest = _manifest()
    save_model_run(manifest, registry, tmp_path / "a.json")
    save_model_run(manifest, registry, tmp_path / "a.json")
    assert len(list_model_runs(registry)) == 1


def test_compare_model_runs_from_manifest_paths(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_manifest(value=0.01)), encoding="utf-8")
    b.write_text(json.dumps(_manifest(value=0.03)), encoding="utf-8")
    result = compare_model_runs(str(a), str(b), registry_path=tmp_path / "runs.csv")
    assert result["winner"] == "b"
    assert result["delta_b_minus_a"] > 0


def test_compare_model_runs_lower_metric_wins(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_manifest(value=-0.01, metric="mae_vs_vol_20_persistence", task="volatility")), encoding="utf-8")
    b.write_text(json.dumps(_manifest(value=-0.03, metric="mae_vs_vol_20_persistence", task="volatility")), encoding="utf-8")
    result = compare_model_runs(str(a), str(b), registry_path=tmp_path / "runs.csv")
    assert result["metric_direction"] == "lower"
    assert result["winner"] == "b"


def test_rank_model_runs_respects_metric_direction(tmp_path):
    registry = tmp_path / "runs.csv"
    save_model_run(_manifest(value=0.01), registry, tmp_path / "a.json")
    save_model_run(_manifest(value=0.03), registry, tmp_path / "b.json")
    rows = rank_model_runs(registry, task="direction")
    assert rows[0]["primary_value"] == 0.03
    assert rows[0]["rank"] == 1

    save_model_run(
        _manifest(value=-0.01, metric="mae_vs_vol_20_persistence", task="volatility"),
        registry,
        tmp_path / "c.json",
    )
    save_model_run(
        _manifest(value=-0.03, metric="mae_vs_vol_20_persistence", task="volatility"),
        registry,
        tmp_path / "d.json",
    )
    rows = rank_model_runs(registry, task="volatility")
    assert rows[0]["primary_value"] == -0.03
    assert rows[0]["rank"] == 1


def test_cli_model_runs_best(capsys, tmp_path):
    registry = tmp_path / "runs.csv"
    save_model_run(_manifest(value=0.01), registry, tmp_path / "a.json")
    save_model_run(_manifest(value=0.03), registry, tmp_path / "b.json")
    code = cmd_model_runs(argparse.Namespace(
        action="best",
        refs=[],
        registry=str(registry),
        limit=1,
        task="direction",
        dataset_id=None,
        primary_metric=None,
    ))
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["runs"]) == 1
    assert out["runs"][0]["primary_value"] == 0.03
