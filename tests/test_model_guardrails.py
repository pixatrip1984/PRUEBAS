import json

from eso.cli import main
from eso.modeling.guardrails import validate_model_run_guardrails


def _manifest():
    return {
        "schema_version": "eso.model_run.v1",
        "dataset_id": "unit",
        "task": "direction",
        "model_type": "logistic",
        "features": ["x"],
        "feature_frame": {"sha256": "a" * 64},
        "target": {"name": "future_direction_3", "type": "classification", "horizon": 3},
        "split": {"train_size": 10, "test_size": 5, "shuffle": False},
        "baselines": {"majority_class_accuracy": 0.5},
        "metrics": {"accuracy_vs_baseline": 0.1},
        "primary_metric": "accuracy_vs_baseline",
        "causal_guardrails": [
            "Temporal split only; shuffle=False.",
            "Target may use future prices only as labels.",
        ],
    }


def test_validate_model_run_guardrails_accepts_valid_manifest():
    assert validate_model_run_guardrails(_manifest()) == []


def test_validate_model_run_guardrails_rejects_shuffle_and_missing_baseline():
    manifest = _manifest()
    manifest["split"]["shuffle"] = True
    manifest["baselines"] = {}
    errors = validate_model_run_guardrails(manifest)
    assert "split.shuffle must be false" in errors
    assert "baselines must not be empty" in errors


def test_cli_validate_model_run(tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert main(["validate-model-run", str(path)]) == 0
    assert '"valid": true' in capsys.readouterr().out

    bad = _manifest()
    bad["features"] = []
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert main(["validate-model-run", str(path)]) == 1
    assert "features must not be empty" in capsys.readouterr().out
