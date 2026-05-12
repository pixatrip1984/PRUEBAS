"""Preprocessing utilities for ESO target datasets."""

from __future__ import annotations

import numpy as np


def clean_numeric(data, fill: str = "median") -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = x.copy()
    x[~np.isfinite(x)] = np.nan
    if fill == "drop":
        return x[~np.isnan(x).any(axis=1)]
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.all(np.isnan(col)):
            x[:, j] = 0.0
            continue
        value = np.nanmedian(col) if fill == "median" else np.nanmean(col)
        col[np.isnan(col)] = value
        x[:, j] = col
    return x


def robust_scale(data, clip: float | None = 8.0) -> np.ndarray:
    x = clean_numeric(data)
    med = np.median(x, axis=0, keepdims=True)
    q1 = np.percentile(x, 25, axis=0, keepdims=True)
    q3 = np.percentile(x, 75, axis=0, keepdims=True)
    scale = np.maximum(q3 - q1, 1e-12)
    y = (x - med) / scale
    return np.clip(y, -clip, clip) if clip is not None else y


def standard_scale(data) -> np.ndarray:
    x = clean_numeric(data)
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.maximum(np.std(x, axis=0, keepdims=True), 1e-12)
    return (x - mu) / sd


def minmax_scale(data) -> np.ndarray:
    x = clean_numeric(data)
    lo = np.min(x, axis=0, keepdims=True)
    hi = np.max(x, axis=0, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def normalize(data, method: str = "robust") -> np.ndarray:
    if method in {None, "none"}:
        return clean_numeric(data)
    if method == "robust":
        return robust_scale(data)
    if method == "standard":
        return standard_scale(data)
    if method == "minmax":
        return minmax_scale(data)
    raise ValueError(f"unknown normalize method: {method}")


def flatten_windows(windows: np.ndarray, mode: str = "last") -> np.ndarray:
    w = np.asarray(windows, dtype=float)
    if w.ndim != 3:
        return w
    if mode == "last":
        return w[:, -1, :]
    if mode == "mean":
        return np.mean(w, axis=1)
    if mode == "flat":
        return w.reshape(w.shape[0], -1)
    raise ValueError(f"unknown window mode: {mode}")
