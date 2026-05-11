"""Numeric signatures for ESO experiments."""

from __future__ import annotations

import math


def _safe_float(value, default=float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def diagnosis_signature(diagnosis: dict) -> dict:
    """Extract a fixed set of scalar features from a blind diagnosis."""
    dim = diagnosis.get("dimension", {})
    stat = diagnosis.get("stationarity", {})
    lyap = diagnosis.get("lyapunov", {})
    sym = diagnosis.get("symmetries", {})
    freq = sym.get("dominant_frequency", {}) if isinstance(sym.get("dominant_frequency", {}), dict) else {}
    label = stat.get("label", "unknown")
    return {
        "dim_consensus": _safe_float(dim.get("consensus_dimension")),
        "dim_two_nn": _safe_float(dim.get("two_nn", {}).get("dimension")),
        "dim_pca": _safe_float(dim.get("pca", {}).get("dimension")),
        "dim_correlation": _safe_float(dim.get("correlation", {}).get("dimension")),
        "lyapunov_max": _safe_float(lyap.get("value")),
        "chaotic_hint": float(bool(lyap.get("chaotic_hint", False))),
        "stationary_hint": float(label == "stationary"),
        "periodic_hint": float(bool(sym.get("periodic_hint", False))),
        "dominant_frequency": _safe_float(freq.get("frequency")),
    }


def experiment_signature(diagnosis: dict, evaluation: dict) -> dict:
    sig = diagnosis_signature(diagnosis)
    sig.update(
        {
            "manifold": evaluation.get("manifold"),
            "manifold_dim": _safe_float(evaluation.get("manifold_dim", float("nan"))),
            "reconstruction_error": _safe_float(evaluation.get("reconstruction_error")),
            "smoothness": _safe_float(evaluation.get("smoothness")),
            "latent_utilization": _safe_float(evaluation.get("latent_utilization")),
        }
    )
    return sig
