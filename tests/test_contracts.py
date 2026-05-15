import json

import pytest

from eso.contracts import assert_valid_contract, validate_contract
from eso.modeling.supervised import write_model_run


def _model_run():
    return {
        "schema_version": "eso.model_run.v1",
        "dataset_id": "unit",
        "task": "direction",
        "model_type": "logistic",
        "features": ["x"],
        "feature_frame": {"rows": 10, "columns": ["x", "close"], "sha256": "a" * 64},
        "target": {"name": "future_direction_1", "type": "classification", "horizon": 1},
        "split": {"train_size": 7, "test_size": 3, "shuffle": False},
        "baselines": {"majority_class_accuracy": 0.5},
        "metrics": {"accuracy_vs_baseline": 0.1},
        "primary_metric": "accuracy_vs_baseline",
        "feature_importances": {"x": 0.1},
        "causal_guardrails": ["Temporal split only; shuffle=False."],
    }


def test_validate_contract_accepts_model_run():
    assert validate_contract(_model_run()) == []


def test_validate_contract_reports_missing_required_key():
    payload = _model_run()
    del payload["split"]
    errors = validate_contract(payload, "eso.model_run.v1")
    assert any("split" in err for err in errors)


def test_write_model_run_validates_contract(tmp_path):
    path = tmp_path / "run.json"
    write_model_run(_model_run(), path)
    assert json.loads(path.read_text())["schema_version"] == "eso.model_run.v1"


def test_assert_valid_contract_raises_on_bad_const():
    payload = _model_run()
    payload["schema_version"] = "bad"
    with pytest.raises(ValueError, match="contract validation"):
        assert_valid_contract(payload, "eso.model_run.v1")


def test_validate_contract_accepts_benchmark_suite():
    payload = {
        "schema_version": "eso.benchmark_suite.v1",
        "name": "unit",
        "dataset": {"path": "data.csv"},
        "runs": [{"name": "vol", "task": "volatility"}],
    }
    assert validate_contract(payload) == []


def test_validate_contract_accepts_asset_universe():
    payload = {
        "schema_version": "eso.asset_universe.v1",
        "name": "unit",
        "assets": [{"asset_id": "BTC", "path": "btc.csv"}],
    }
    assert validate_contract(payload) == []
