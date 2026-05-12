"""Evaluation metrics for manifold tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .manifolds import Manifold, Plane, get_manifold
from .reconstruction import masked_neighbor_reconstruction, project_data_to_manifold

# Cache flat-baseline errors per (data_hash, ambient_dim, k, mask_ratio, seed)
# to avoid re-running plane baselines for every manifold in a batch.
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
) -> float:
    """Reconstruction error of a flat Plane(ambient_dim) on the same data+seed.

    Cached so repeated calls within a batch are free.
    """
    cache_key = (id(data), data.shape, ambient_dim, k, mask_ratio, seed)
    if cache_key not in _plane_baseline_cache:
        plane = Plane(ambient_dim)
        latent = project_data_to_manifold(data, plane)
        result = masked_neighbor_reconstruction(data, latent, plane, k=k, mask_ratio=mask_ratio, seed=seed)
        _plane_baseline_cache[cache_key] = result.reconstruction_error
    return _plane_baseline_cache[cache_key]


def evaluate_manifold(
    data,
    manifold: str | Manifold,
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
) -> dict:
    mani = get_manifold(manifold) if isinstance(manifold, str) else manifold
    x = np.asarray(data, dtype=float)
    latent = project_data_to_manifold(x, mani)
    result = masked_neighbor_reconstruction(x, latent, mani, k=k, mask_ratio=mask_ratio, seed=seed)
    score, penalty = _score(result.reconstruction_error, mani.dim, mani.ambient_dim, complexity_weight)

    # Normalised metric: how much worse (or better) than a flat space of the
    # same ambient dimension on the same data?  Values < 1.0 mean the curved
    # manifold beats flat — genuine geometric structure.
    flat_err = _flat_baseline_error(x, mani.ambient_dim, k, mask_ratio, seed)
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
    return out


def rank_manifolds(
    data,
    manifolds: list[str],
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
) -> list[dict]:
    results = [
        evaluate_manifold(data, m, k=k, mask_ratio=mask_ratio, seed=seed, complexity_weight=complexity_weight)
        for m in manifolds
    ]
    return sorted(results, key=lambda r: r["score"] if np.isfinite(r["score"]) else np.inf)
