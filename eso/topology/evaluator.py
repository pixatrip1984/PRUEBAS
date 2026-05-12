"""Evaluation metrics for manifold tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .manifolds import Manifold, Plane, get_manifold
from .reconstruction import (
    clear_embed_cache,
    masked_neighbor_reconstruction,
    project_data_to_manifold,
)

# Flat-baseline cache: (data_id, shape, ambient_dim, method, seed, k, mask_ratio)
_plane_baseline_cache: dict = {}


@dataclass
class ManifoldEvaluation:
    manifold: str
    manifold_dim: int
    ambient_dim: int
    reconstruction_error: float
    smoothness: float
    latent_utilization: float
    complexity_penalty: float
    score: float
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


def _score(error: float, manifold_dim: int, ambient_dim: int, complexity_weight: float) -> tuple[float, float]:
    penalty = complexity_weight * (float(manifold_dim) + 0.25 * float(ambient_dim))
    base = error if np.isfinite(error) else np.inf
    return float(base + penalty), float(penalty)


def _flat_baseline_error(
    data: np.ndarray,
    ambient_dim: int,
    k: int,
    mask_ratio: float,
    seed: int | None,
    projection_method: str = "linear",
    proj_neighbors: int = 15,
) -> float:
    """Reconstruction error for Plane(ambient_dim) on the same data/seed/method.

    The flat baseline is method-matched: a curved manifold is only interesting
    when it beats the flat space under the SAME embedding method.
    """
    cache_key = (id(data), data.shape, ambient_dim, k, mask_ratio, seed,
                 projection_method, proj_neighbors)
    if cache_key not in _plane_baseline_cache:
        plane = Plane(ambient_dim)
        latent = project_data_to_manifold(
            data, plane, method=projection_method, seed=seed,
            n_neighbors=proj_neighbors,
        )
        result = masked_neighbor_reconstruction(data, latent, plane, k=k,
                                                mask_ratio=mask_ratio, seed=seed)
        _plane_baseline_cache[cache_key] = result.reconstruction_error
    return _plane_baseline_cache[cache_key]


def evaluate_manifold(
    data,
    manifold: str | Manifold,
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
    projection_method: str = "linear",
    proj_neighbors: int = 15,
    proj_min_dist: float = 0.1,
) -> dict:
    mani = get_manifold(manifold) if isinstance(manifold, str) else manifold
    x = np.asarray(data, dtype=float)
    latent = project_data_to_manifold(
        x, mani,
        method=projection_method,
        seed=seed,
        n_neighbors=proj_neighbors,
        min_dist=proj_min_dist,
    )
    result = masked_neighbor_reconstruction(x, latent, mani, k=k, mask_ratio=mask_ratio, seed=seed)
    score, penalty = _score(result.reconstruction_error, mani.dim, mani.ambient_dim, complexity_weight)

    flat_err = _flat_baseline_error(
        x, mani.ambient_dim, k, mask_ratio, seed,
        projection_method=projection_method, proj_neighbors=proj_neighbors,
    )
    relative_to_flat = (
        float(result.reconstruction_error / flat_err) if flat_err > 0 else float("nan")
    )

    eval_result = ManifoldEvaluation(
        manifold=mani.name,
        manifold_dim=int(mani.dim),
        ambient_dim=int(mani.ambient_dim),
        reconstruction_error=result.reconstruction_error,
        smoothness=graph_smoothness(x, result.edge_index),
        latent_utilization=latent_utilization(latent),
        complexity_penalty=penalty,
        score=score,
        n_points=int(len(x)),
        k=int(k),
        mask_ratio=float(mask_ratio),
    )
    out = eval_result.to_dict()
    out["latent_shape"] = list(latent.shape)
    out["flat_baseline_error"] = float(flat_err)
    out["relative_to_flat"] = relative_to_flat
    out["projection_method"] = projection_method
    return out


def rank_manifolds(
    data,
    manifolds: list[str],
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
    projection_method: str = "linear",
    proj_neighbors: int = 15,
) -> list[dict]:
    results = [
        evaluate_manifold(
            data, m, k=k, mask_ratio=mask_ratio, seed=seed,
            complexity_weight=complexity_weight,
            projection_method=projection_method,
            proj_neighbors=proj_neighbors,
        )
        for m in manifolds
    ]
    return sorted(results, key=lambda r: r["score"] if np.isfinite(r["score"]) else np.inf)
