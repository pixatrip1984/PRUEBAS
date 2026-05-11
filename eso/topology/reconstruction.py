"""Masked relational reconstruction over candidate manifolds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .manifolds import Manifold


@dataclass
class ReconstructionResult:
    reconstruction_error: float
    predictions: np.ndarray
    mask: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray


def masked_neighbor_reconstruction(
    values,
    latent_points,
    manifold: Manifold,
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
) -> ReconstructionResult:
    """Reconstruct masked values from k-nearest latent neighbors.

    This first ESO implementation is deliberately simple and deterministic:
    predictions are inverse-distance weighted averages in the tested manifold's
    ambient coordinates. It is a baseline metric, not the final neural model.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    z = np.asarray(latent_points, dtype=float)
    if len(y) != len(z):
        raise ValueError("values and latent_points must have the same length")
    n = len(y)
    if n < 3:
        return ReconstructionResult(float("nan"), y.copy(), np.ones(n, dtype=bool), np.zeros((2, 0), dtype=int), np.array([]))

    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(z)
    dist, idx = nn.kneighbors(z)
    nbr_idx = idx[:, 1:]
    nbr_dist = dist[:, 1:]
    weights = 1.0 / np.maximum(nbr_dist, 1e-8)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    pred = np.einsum("nk,nkd->nd", weights, y[nbr_idx])

    mask = rng.random(n) < mask_ratio
    if not mask.any():
        mask[rng.integers(0, n)] = True
    err = float(np.mean((pred[mask] - y[mask]) ** 2))

    src = np.repeat(np.arange(n), k_eff - 1)
    dst = nbr_idx.reshape(-1)
    return ReconstructionResult(
        reconstruction_error=err,
        predictions=pred,
        mask=mask,
        edge_index=np.stack([src, dst], axis=0),
        edge_weight=nbr_dist.reshape(-1),
    )


def project_data_to_manifold(data, manifold: Manifold) -> np.ndarray:
    """Project data to the manifold's ambient coordinates using PCA-like truncation.

    This is a non-neural bootstrap projection for the first incremental version.
    TopoVAE can replace this function without changing evaluator interfaces.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x)
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    coords = u[:, : manifold.ambient_dim] * s[: manifold.ambient_dim]
    if coords.shape[1] < manifold.ambient_dim:
        coords = np.pad(coords, [(0, 0), (0, manifold.ambient_dim - coords.shape[1])])
    return manifold.project(coords)
