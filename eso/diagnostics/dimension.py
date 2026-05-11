"""Intrinsic-dimension diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class DimensionEstimate:
    method: str
    dimension: float
    details: dict


def _as_2d(data) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x.reshape(x.shape[0], -1)


def estimate_twonn_dimension(data, min_samples: int = 8) -> DimensionEstimate:
    """Estimate intrinsic dimension with the Two-NN method."""
    x = _as_2d(data)
    if len(x) < min_samples:
        return DimensionEstimate("two_nn", float("nan"), {"reason": "not_enough_samples"})

    nn = NearestNeighbors(n_neighbors=3).fit(x)
    distances, _ = nn.kneighbors(x)
    r1 = np.maximum(distances[:, 1], 1e-12)
    r2 = np.maximum(distances[:, 2], 1e-12)
    mu = np.sort(r2 / r1)
    n = len(mu)
    y = -np.log(1.0 - (np.arange(1, n + 1) - 0.5) / n)
    slope = np.polyfit(np.log(mu), y, deg=1)[0]
    return DimensionEstimate("two_nn", float(max(slope, 0.0)), {"n": int(n)})


def estimate_pca_dimension(data, variance_threshold: float = 0.95) -> DimensionEstimate:
    """Estimate dimension as PCA components needed for cumulative variance."""
    x = _as_2d(data)
    if len(x) < 2:
        return DimensionEstimate("pca", float("nan"), {"reason": "not_enough_samples"})
    pca = PCA().fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dim = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return DimensionEstimate(
        "pca",
        float(dim),
        {"variance_threshold": variance_threshold, "explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
    )


def estimate_correlation_dimension(data, n_radii: int = 24) -> DimensionEstimate:
    """Grassberger-Procaccia style correlation dimension estimate."""
    x = _as_2d(data)
    if len(x) < 16:
        return DimensionEstimate("correlation", float("nan"), {"reason": "not_enough_samples"})

    d = pdist(x)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 16:
        return DimensionEstimate("correlation", float("nan"), {"reason": "degenerate_distances"})

    lo, hi = np.percentile(d, [5, 70])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return DimensionEstimate("correlation", float("nan"), {"reason": "bad_radius_range"})

    radii = np.geomspace(lo, hi, n_radii)
    corr = np.array([(d < r).mean() for r in radii])
    valid = (corr > 0) & (corr < 1)
    if valid.sum() < 4:
        return DimensionEstimate("correlation", float("nan"), {"reason": "insufficient_scaling_region"})

    log_r = np.log(radii[valid])
    log_c = np.log(corr[valid])
    slopes = []
    window = max(4, min(8, valid.sum()))
    for start in range(0, len(log_r) - window + 1):
        s = np.polyfit(log_r[start : start + window], log_c[start : start + window], deg=1)[0]
        if np.isfinite(s):
            slopes.append(s)
    dim = float(np.median(slopes)) if slopes else float(np.polyfit(log_r, log_c, deg=1)[0])
    return DimensionEstimate(
        "correlation",
        float(max(dim, 0.0)),
        {"radii": radii.tolist(), "correlation_sum": corr.tolist()},
    )


def estimate_dimension(data) -> dict:
    """Run several intrinsic-dimension estimates and return a stable summary."""
    two_nn = estimate_twonn_dimension(data)
    pca = estimate_pca_dimension(data)
    corr = estimate_correlation_dimension(data)
    values = [e.dimension for e in (two_nn, pca, corr) if np.isfinite(e.dimension)]
    consensus = float(np.median(values)) if values else float("nan")
    return {
        "consensus_dimension": consensus,
        "two_nn": two_nn.__dict__,
        "pca": pca.__dict__,
        "correlation": corr.__dict__,
    }
