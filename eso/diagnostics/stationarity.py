"""Stationarity and recurrence diagnostics."""

from __future__ import annotations

import numpy as np

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover
    adfuller = None
    kpss = None


def _as_series(data) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    return np.asarray(x, dtype=float).ravel()


def adf_test(data) -> dict:
    x = _as_series(data)
    if adfuller is None or len(x) < 12:
        return {"available": False, "pvalue": float("nan")}
    try:
        stat, pvalue, usedlag, nobs, critical, _ = adfuller(x, autolag="AIC")
        return {"available": True, "statistic": float(stat), "pvalue": float(pvalue), "used_lag": int(usedlag), "nobs": int(nobs), "critical_values": critical}
    except Exception as exc:
        return {"available": False, "error": str(exc), "pvalue": float("nan")}


def kpss_test(data) -> dict:
    x = _as_series(data)
    if kpss is None or len(x) < 12:
        return {"available": False, "pvalue": float("nan")}
    try:
        stat, pvalue, usedlag, critical = kpss(x, regression="c", nlags="auto")
        return {"available": True, "statistic": float(stat), "pvalue": float(pvalue), "used_lag": int(usedlag), "critical_values": critical}
    except Exception as exc:
        return {"available": False, "error": str(exc), "pvalue": float("nan")}


def recurrence_plot(data, threshold_quantile: float = 0.1, max_points: int = 512) -> np.ndarray:
    """Return a simple binary recurrence matrix."""
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    eps = np.quantile(dist[dist > 0], threshold_quantile) if np.any(dist > 0) else 0.0
    return (dist <= eps).astype(np.uint8)


def stationarity_report(data) -> dict:
    adf = adf_test(data)
    kp = kpss_test(data)
    adf_stationary = adf.get("available") and adf.get("pvalue", 1.0) < 0.05
    kpss_stationary = kp.get("available") and kp.get("pvalue", 0.0) > 0.05

    if adf_stationary and kpss_stationary:
        label = "stationary"
    elif not adf_stationary and not kpss_stationary:
        label = "trend_or_unit_root"
    else:
        label = "ambiguous_or_structural_change"

    return {"label": label, "adf": adf, "kpss": kp}
