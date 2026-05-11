"""Integrated blind diagnosis report."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .dimension import estimate_dimension
from .lyapunov import max_lyapunov_exponent
from .stationarity import stationarity_report
from .symmetries import periodicity_report


def _json_safe(obj: Any):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def summarize_diagnosis(report: dict) -> str:
    dim = report.get("dimension", {}).get("consensus_dimension", float("nan"))
    stationarity = report.get("stationarity", {}).get("label", "unknown")
    lyap = report.get("lyapunov", {}).get("value", float("nan"))
    chaotic = report.get("lyapunov", {}).get("chaotic_hint", False)
    periodic = report.get("symmetries", {}).get("periodic_hint", False)
    freq = report.get("symmetries", {}).get("dominant_frequency", {}).get("frequency", float("nan"))

    parts = [
        f"intrinsic_dimension≈{dim:.3f}" if np.isfinite(dim) else "intrinsic_dimension=unknown",
        f"stationarity={stationarity}",
        f"lyapunov={lyap:.4f}" if np.isfinite(lyap) else "lyapunov=unknown",
        f"chaotic_hint={bool(chaotic)}",
        f"periodic_hint={bool(periodic)}",
    ]
    if np.isfinite(freq):
        parts.append(f"dominant_frequency={freq:.4f}")
    return "; ".join(parts)


def run_diagnosis(data, sample_rate: float = 1.0) -> dict:
    """Run all blind diagnostics and return a structured report."""
    report = {
        "dimension": estimate_dimension(data),
        "stationarity": stationarity_report(data),
        "lyapunov": max_lyapunov_exponent(data),
        "symmetries": periodicity_report(data),
    }
    report["summary"] = summarize_diagnosis(report)
    return _json_safe(report)


def diagnosis_to_json(report: dict, indent: int = 2) -> str:
    return json.dumps(_json_safe(report), indent=indent, sort_keys=True)
