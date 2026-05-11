"""Lyapunov diagnostics for time series."""

from __future__ import annotations

import numpy as np

try:
    import nolds
except Exception:  # pragma: no cover
    nolds = None


def _as_series(data) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    return x.ravel()


def max_lyapunov_exponent(data, emb_dim: int = 6, lag: int = 1, min_tsep: int | None = None) -> dict:
    """Estimate the largest Lyapunov exponent.

    Uses nolds.lyap_r when available. Returns a conservative structured result
    instead of raising on short or ill-conditioned series.
    """
    x = _as_series(data)
    if len(x) < max(64, emb_dim * lag * 4):
        return {"available": False, "value": float("nan"), "reason": "not_enough_samples"}
    if nolds is None:
        return {"available": False, "value": float("nan"), "reason": "nolds_not_installed"}
    try:
        value = nolds.lyap_r(x, emb_dim=emb_dim, lag=lag, min_tsep=min_tsep, fit="poly")
        return {"available": True, "value": float(value), "chaotic_hint": bool(value > 0.0)}
    except Exception as exc:
        return {"available": False, "value": float("nan"), "error": str(exc)}
