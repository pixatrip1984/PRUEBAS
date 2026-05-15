"""Run declarative supervised benchmark suites."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eso.contracts import assert_valid_contract
from eso.data.features import prepare_btc_klines
from eso.data.loader import read_table
from eso.modeling import build_feature_frame, evaluate_supervised_task, save_model_run, write_model_run
from eso.modeling.registry import metric_direction


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass
    return obj


def _slice_rows(df: pd.DataFrame, skip_rows: int | None, max_rows: int | None) -> pd.DataFrame:
    if skip_rows:
        df = df.iloc[int(skip_rows):]
    if max_rows:
        df = df.head(int(max_rows))
    return df.reset_index(drop=True)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_") or "run"


def _load_suite(suite: str | Path | dict) -> dict:
    if isinstance(suite, dict):
        payload = suite
    else:
        payload = json.loads(Path(suite).read_text(encoding="utf-8"))
    assert_valid_contract(payload, "eso.benchmark_suite.v1")
    return payload


def _run_summary(manifest: dict, run_id: str | None, path: str) -> dict:
    primary = manifest["primary_metric"]
    value = manifest["metrics"].get(primary)
    return {
        "name": manifest.get("run_name"),
        "run_id": run_id,
        "task": manifest["task"],
        "model_type": manifest["model_type"],
        "target": manifest["target"],
        "split": manifest["split"],
        "features": manifest["features"],
        "primary_metric": primary,
        "primary_value": value,
        "metric_direction": metric_direction(primary),
        "manifest_path": path,
    }


def run_benchmark_suite(
    suite: str | Path | dict,
    output_dir: str | Path,
    registry_path: str | Path = "reports/model_runs.csv",
    save_registry: bool = True,
) -> dict:
    """Execute a benchmark suite and return a versioned summary."""
    spec = _load_suite(suite)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = spec["dataset"]
    raw = read_table(dataset["path"])
    if dataset.get("klines_format", False):
        raw = prepare_btc_klines(raw)
    raw = _slice_rows(raw, dataset.get("skip_rows"), dataset.get("max_rows"))

    feature_mode = dataset.get("feature_mode", "raw")
    feature_frame = build_feature_frame(
        raw,
        feature_mode=feature_mode,
        columns=dataset.get("columns"),
        close_col=dataset.get("close_col", "close"),
        umap_train_size=dataset.get("umap_train_size", 27000),
        smooth_windows=tuple(dataset.get("smooth_windows", [6, 24, 72])),
    )

    runs = []
    errors = []
    for run in spec.get("runs", []):
        name = run["name"]
        run_dir = output_dir / _safe_name(name)
        manifest_path = run_dir / "model_run_manifest.json"
        try:
            manifest = evaluate_supervised_task(
                feature_frame,
                task=run["task"],
                horizon=run.get("horizon", 12),
                feature_cols=run.get("feature_columns"),
                train_size=run.get("train_size", spec.get("train_size")),
                train_fraction=run.get("train_fraction", spec.get("train_fraction", 0.7)),
                model_type=run.get("model_type", "auto"),
                dataset_id=spec.get("dataset_id", Path(dataset["path"]).stem),
            )
            manifest["benchmark_suite"] = spec.get("name")
            manifest["run_name"] = name
            write_model_run(manifest, manifest_path)
            row = save_model_run(manifest, registry_path, manifest_path) if save_registry else None
            runs.append(_run_summary(manifest, None if row is None else row["run_id"], str(manifest_path)))
        except Exception as exc:
            errors.append({"name": name, "error": str(exc)})

    result = {
        "schema_version": "eso.benchmark_result.v1",
        "suite_name": spec.get("name"),
        "dataset_id": spec.get("dataset_id", Path(dataset["path"]).stem),
        "dataset": dataset,
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "error_count": len(errors),
        "runs": runs,
        "errors": errors,
        "guardrails": [
            "Each run uses the shared ESO target builders.",
            "Each run uses a temporal split; shuffle=False.",
            "Each run writes a schema-validated model_run_manifest.json.",
        ],
    }
    if save_registry:
        result["registry_path"] = str(registry_path)
    assert_valid_contract(result, "eso.benchmark_result.v1")
    return result


def write_benchmark_result(result: dict, output_path: str | Path) -> str:
    assert_valid_contract(result, "eso.benchmark_result.v1")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
