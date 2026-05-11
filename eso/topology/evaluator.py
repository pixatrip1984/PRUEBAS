"""Evaluation metrics for manifold tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .manifolds import Manifold, get_manifold
from .reconstruction import masked_neighbor_reconstruction, project_data_to_manifold


@dataclass
class ManifoldEvaluation:
    manifold: str
    reconstruction_error: float
    smoothness: float
    latent_utilization: float
    n_points: int
    k: int
    mask_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def graph_smoothness(values, edge_index: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if edge_index.size == 0:
        return float("nan")
    src, dst = edge_index
    return float(np.mean(np.sum((y[src] - y[dst]) ** 2, axis=1)))


def latent_utilization(latent_points, bins: int = 12) -> float:
    """Coarse occupancy ratio in latent ambient coordinates."""
    z = np.asarray(latent_points, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if len(z) < 2:
        return float("nan")
    mins = z.min(axis=0)
    maxs = z.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    scaled = np.clip((z - mins) / span, 0.0, 0.999999)
    idx = np.floor(scaled * bins).astype(int)
    occupied = len({tuple(row) for row in idx})
    possible = min(len(z), bins ** z.shape[1])
    return float(occupied / max(possible, 1))


def evaluate_manifold(data, manifold: str | Manifold, k: int = 8, mask_ratio: float = 0.25, seed: int | None = None) -> dict:
    mani = get_manifold(manifold) if isinstance(manifold, str) else manifold
    x = np.asarray(data, dtype=float)
    latent = project_data_to_manifold(x, mani)
    result = masked_neighbor_reconstruction(x, latent, mani, k=k, mask_ratio=mask_ratio, seed=seed)
    eval_result = ManifoldEvaluation(
        manifold=mani.name,
        reconstruction_error=result.reconstruction_error,
        smoothness=graph_smoothness(x, result.edge_index),
        latent_utilization=latent_utilization(latent),
        n_points=int(len(x)),
        k=int(k),
        mask_ratio=float(mask_ratio),
    )
    out = eval_result.to_dict()
    out["latent_shape"] = list(latent.shape)
    return out


def rank_manifolds(data, manifolds: list[str], k: int = 8, mask_ratio: float = 0.25, seed: int | None = None) -> list[dict]:
    results = [evaluate_manifold(data, m, k=k, mask_ratio=mask_ratio, seed=seed) for m in manifolds]
    return sorted(results, key=lambda r: r["reconstruction_error"] if np.isfinite(r["reconstruction_error"]) else np.inf)
