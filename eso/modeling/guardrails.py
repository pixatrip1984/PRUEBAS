"""Validation guardrails for model-run manifests."""

from __future__ import annotations

from pathlib import Path
import json


def validate_model_run_guardrails(manifest: dict) -> list[str]:
    """Return human-readable model-run guardrail violations."""
    errors: list[str] = []
    split = manifest.get("split") or {}
    if split.get("shuffle") is not False:
        errors.append("split.shuffle must be false")
    if int(split.get("train_size") or 0) <= 0:
        errors.append("split.train_size must be positive")
    if int(split.get("test_size") or 0) <= 0:
        errors.append("split.test_size must be positive")

    task = manifest.get("task")
    baselines = manifest.get("baselines") or {}
    if not baselines:
        errors.append("baselines must not be empty")
    elif task == "volatility" and not any(k in baselines for k in ("vol_20_persistence_mae", "train_mean_mae")):
        errors.append("volatility manifest needs a volatility baseline")
    elif task in {"regime", "direction"} and "majority_class_accuracy" not in baselines:
        errors.append(f"{task} manifest needs majority_class_accuracy baseline")

    primary = manifest.get("primary_metric")
    metrics = manifest.get("metrics") or {}
    if not primary:
        errors.append("primary_metric is required")
    elif primary not in metrics or metrics.get(primary) is None:
        errors.append(f"primary_metric value missing in metrics: {primary}")

    feature_frame = manifest.get("feature_frame") or {}
    if not feature_frame.get("sha256"):
        errors.append("feature_frame.sha256 is required")
    if not manifest.get("features"):
        errors.append("features must not be empty")

    guardrails = manifest.get("causal_guardrails") or []
    guardrail_text = " ".join(str(item).lower() for item in guardrails)
    if "temporal" not in guardrail_text or "shuffle=false" not in guardrail_text.replace(" ", ""):
        errors.append("causal_guardrails must explicitly mention temporal split and shuffle=False")
    if "future prices" not in guardrail_text and "future" not in guardrail_text:
        errors.append("causal_guardrails must mention future labels/prices")
    return errors


def validate_model_run_file(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_model_run_guardrails(payload)
    return {"path": str(path), "valid": not errors, "errors": errors}
