"""Simple symmetry, autocorrelation and spectral diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _as_series(data) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.ravel()
    return x[np.isfinite(x)]


def autocorrelation(data, max_lag: int | None = None) -> np.ndarray:
    x = _as_series(data)
    if len(x) < 3:
        return np.array([], dtype=float)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 0:
        return np.ones(1, dtype=float)
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :] / denom
    if max_lag is not None:
        corr = corr[: max_lag + 1]
    return corr


def dominant_frequency(data, sample_rate: float = 1.0) -> dict:
    x = _as_series(data)
    if len(x) < 8:
        return {"frequency": float("nan"), "power": float("nan"), "available": False}
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(freqs) <= 1:
        return {"frequency": float("nan"), "power": float("nan"), "available": False}
    idx = int(np.argmax(power[1:]) + 1)
    return {"frequency": float(freqs[idx]), "power": float(power[idx]), "available": True}


def periodicity_report(data, max_lag: int | None = None) -> dict:
    x = _as_series(data)
    if max_lag is None:
        max_lag = max(2, min(len(x) // 2, 256))
    corr = autocorrelation(x, max_lag=max_lag)
    if corr.size < 4:
        return {"periodic_hint": False, "peaks": [], "autocorrelation": corr.tolist()}
    peaks, props = find_peaks(corr[1:], height=0.2)
    peaks = peaks + 1
    return {
        "periodic_hint": bool(len(peaks) > 0),
        "peaks": peaks.astype(int).tolist(),
        "peak_heights": props.get("peak_heights", np.array([])).astype(float).tolist(),
        "dominant_frequency": dominant_frequency(x),
        "autocorrelation": corr.tolist(),
    }
