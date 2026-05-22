"""Registry for supervised model-run manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REGISTRY_COLUMNS = [
    "run_id",
    "created_at",
    "dataset_id",
    "task",
    "target_name",
    "target_type",
    "horizon",
    "model_type",
    "train_size",
    "test_size",
    "feature_count",
    "feature_sha256",
    "primary_metric",
    "primary_value",
    "metric_direction",
    "metrics_json",
    "manifest_path",
]


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def model_run_id(manifest: dict) -> str:
    """Return a stable id for a model-run manifest."""
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()[:16]


def metric_direction(metric: str) -> str:
    """Return whether higher or lower values are better for a metric."""
    metric = metric.lower()
    if metric.startswith(("mae", "rmse", "loss", "error")):
        return "lower"
    if metric in {"accuracy", "accuracy_vs_baseline", "pearson_r", "r2"}:
        return "higher"
    if metric.endswith("_r"):
        return "higher"
    return "higher"


def summarize_manifest(manifest: dict, manifest_path: str | None = None) -> dict:
    """Convert a full manifest into one registry row."""
    primary = manifest.get("primary_metric")
    metrics = manifest.get("metrics", {})
    primary_value = metrics.get(primary) if primary else None
    target = manifest.get("target", {})
    split = manifest.get("split", {})
    features = manifest.get("features", [])
    feature_frame = manifest.get("feature_frame", {})
    return {
        "run_id": model_run_id(manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest.get("dataset_id"),
        "task": manifest.get("task"),
        "target_name": target.get("name"),
        "target_type": target.get("type"),
        "horizon": target.get("horizon"),
        "model_type": manifest.get("model_type"),
        "train_size": split.get("train_size"),
        "test_size": split.get("test_size"),
        "feature_count": len(features),
        "feature_sha256": feature_frame.get("sha256"),
        "primary_metric": primary,
        "primary_value": primary_value,
        "metric_direction": metric_direction(str(primary)),
        "metrics_json": json.dumps(metrics, sort_keys=True),
        "manifest_path": manifest_path,
    }


def _read_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    df = pd.read_csv(path)
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[REGISTRY_COLUMNS]


def save_model_run(
    manifest: dict,
    registry_path: str | Path = "reports/model_runs.csv",
    manifest_path: str | Path | None = None,
) -> dict:
    """Add or replace one manifest summary in the registry."""
    registry_path = Path(registry_path)
    row = summarize_manifest(manifest, str(manifest_path) if manifest_path else None)
    df = _read_registry(registry_path)
    df = df[df["run_id"] != row["run_id"]]
    row_df = pd.DataFrame([row], columns=REGISTRY_COLUMNS)
    df = row_df if df.empty else pd.concat([df, row_df], ignore_index=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(registry_path, index=False)
    return row


def list_model_runs(
    registry_path: str | Path = "reports/model_runs.csv",
    limit: int | None = None,
    task: str | None = None,
    dataset_id: str | None = None,
) -> list[dict]:
    """Return registry rows, newest first."""
    df = _read_registry(registry_path)
    if task:
        df = df[df["task"] == task]
    if dataset_id:
        df = df[df["dataset_id"] == dataset_id]
    df = df.sort_values("created_at", ascending=False, na_position="last")
    if limit:
        df = df.head(int(limit))
    return df.to_dict(orient="records")


def rank_model_runs(
    registry_path: str | Path = "reports/model_runs.csv",
    limit: int | None = None,
    task: str | None = None,
    dataset_id: str | None = None,
    primary_metric: str | None = None,
) -> list[dict]:
    """Return model runs ranked by their primary metric direction."""
    df = _read_registry(registry_path)
    if task:
        df = df[df["task"] == task]
    if dataset_id:
        df = df[df["dataset_id"] == dataset_id]
    if primary_metric:
        df = df[df["primary_metric"] == primary_metric]
    if df.empty:
        return []

    df = df.copy()
    df["primary_value"] = pd.to_numeric(df["primary_value"], errors="coerce")
    df = df.dropna(subset=["primary_value"])
    if df.empty:
        return []

    ranked_frames = []
    for (_metric, direction), grp in df.groupby(["primary_metric", "metric_direction"], dropna=False):
        ascending = str(direction) == "lower"
        grp = grp.sort_values("primary_value", ascending=ascending).copy()
        best = float(grp.iloc[0]["primary_value"])
        if ascending:
            grp["delta_from_best"] = grp["primary_value"] - best
        else:
            grp["delta_from_best"] = best - grp["primary_value"]
        grp["rank"] = range(1, len(grp) + 1)
        ranked_frames.append(grp)

    ranked = pd.concat(ranked_frames, ignore_index=True)
    ranked = ranked.sort_values(
        ["primary_metric", "rank", "created_at"],
        ascending=[True, True, False],
        na_position="last",
    )
    if limit:
        ranked = ranked.head(int(limit))
    return ranked.to_dict(orient="records")


def _load_manifest_or_row(ref: str, registry_path: str | Path) -> dict:
    path = Path(ref)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = list_model_runs(registry_path)
    for row in rows:
        if row.get("run_id") == ref:
            manifest_path = row.get("manifest_path")
            if manifest_path and Path(str(manifest_path)).exists():
                return json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
            return {
                "dataset_id": row.get("dataset_id"),
                "task": row.get("task"),
                "model_type": row.get("model_type"),
                "target": {
                    "name": row.get("target_name"),
                    "type": row.get("target_type"),
                    "horizon": row.get("horizon"),
                },
                "split": {
                    "train_size": row.get("train_size"),
                    "test_size": row.get("test_size"),
                },
                "features": [None] * int(row.get("feature_count") or 0),
                "feature_frame": {"sha256": row.get("feature_sha256")},
                "primary_metric": row.get("primary_metric"),
                "metrics": json.loads(row.get("metrics_json") or "{}"),
            }
    raise ValueError(f"model run not found: {ref}")


def compare_model_runs(ref_a: str, ref_b: str, registry_path: str | Path = "reports/model_runs.csv") -> dict:
    """Compare two model runs by their primary metric."""
    a = _load_manifest_or_row(ref_a, registry_path)
    b = _load_manifest_or_row(ref_b, registry_path)
    metric_a = a.get("primary_metric")
    metric_b = b.get("primary_metric")
    if metric_a != metric_b:
        raise ValueError(f"primary metrics differ: {metric_a} vs {metric_b}")
    metric = str(metric_a)
    direction = metric_direction(metric)
    value_a = a.get("metrics", {}).get(metric)
    value_b = b.get("metrics", {}).get(metric)
    if value_a is None or value_b is None:
        raise ValueError(f"missing primary metric value: {metric}")
    delta = float(value_b) - float(value_a)
    if direction == "higher":
        winner = "b" if delta > 0 else "a" if delta < 0 else "tie"
    else:
        winner = "b" if delta < 0 else "a" if delta > 0 else "tie"
    return {
        "metric": metric,
        "metric_direction": direction,
        "a": {"ref": ref_a, "value": float(value_a), "run_id": model_run_id(a)},
        "b": {"ref": ref_b, "value": float(value_b), "run_id": model_run_id(b)},
        "delta_b_minus_a": delta,
        "winner": winner,
    }
