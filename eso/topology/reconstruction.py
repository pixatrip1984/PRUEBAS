"""Masked relational reconstruction over candidate manifolds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .manifolds import Manifold

# Cache keyed by (data object id, shape, n_components, method, seed, n_neighbors)
# so that multiple manifolds sharing the same ambient_dim reuse the same embedding.
_embed_cache: dict = {}


@dataclass
class ReconstructionResult:
    reconstruction_error: float
    predictions: np.ndarray
    mask: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray


def _svd_embed(x: np.ndarray, n_components: int) -> np.ndarray:
    u, s, _vh = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    if coords.shape[1] < n_components:
        coords = np.pad(coords, [(0, 0), (0, n_components - coords.shape[1])])
    return coords


def embed_data(
    data: np.ndarray,
    n_components: int,
    method: str = "linear",
    seed: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> np.ndarray:
    """Reduce data to n_components dimensions using the specified method.

    Results are cached per (data identity, shape, n_components, method, seed,
    n_neighbors) so that all manifolds with the same ambient_dim share one
    embedding run — critical for UMAP/Isomap which are expensive.

    Args:
        data:        (n, d) float array, will be centred in-place before embedding.
        n_components: target dimensionality (= manifold.ambient_dim).
        method:      'linear' | 'umap' | 'isomap'.
        seed:        random seed (for UMAP reproducibility).
        n_neighbors: neighbourhood size for UMAP and Isomap.
        min_dist:    UMAP min_dist parameter.
        metric:      distance metric.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    cache_key = (id(data), data.shape if hasattr(data, "shape") else x.shape,
                 n_components, method, seed, n_neighbors)
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    xc = x - np.nanmean(x, axis=0, keepdims=True)
    xc = np.nan_to_num(xc)
    n_eff = min(n_components, xc.shape[1])

    if method == "linear":
        coords = _svd_embed(xc, n_components)

    elif method == "umap":
        try:
            import umap as umap_lib
        except ImportError as e:
            raise ImportError("umap-learn is required: pip install umap-learn") from e
        reducer = umap_lib.UMAP(
            n_components=n_eff,
            n_neighbors=min(n_neighbors, len(xc) - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            verbose=False,
        )
        coords = reducer.fit_transform(xc)
        if coords.shape[1] < n_components:
            coords = np.pad(coords, [(0, 0), (0, n_components - coords.shape[1])])

    elif method == "isomap":
        from sklearn.manifold import Isomap
        iso = Isomap(
            n_components=n_eff,
            n_neighbors=min(n_neighbors, len(xc) - 1),
            metric=metric,
        )
        coords = iso.fit_transform(xc)
        if coords.shape[1] < n_components:
            coords = np.pad(coords, [(0, 0), (0, n_components - coords.shape[1])])

    else:
        raise ValueError(f"Unknown projection method: '{method}'. Choose: linear, umap, isomap")

    _embed_cache[cache_key] = coords
    return coords


def clear_embed_cache() -> None:
    """Evict all cached embeddings (call between datasets to free memory)."""
    _embed_cache.clear()


def project_data_to_manifold(
    data,
    manifold: Manifold,
    method: str = "linear",
    seed: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project data to manifold ambient coordinates then snap to the manifold surface.

    The embedding step reduces data to manifold.ambient_dim dimensions via
    the chosen method (linear SVD, UMAP, or Isomap), then manifold.project()
    snaps the coordinates onto the manifold surface.
    """
    x = np.asarray(data, dtype=float)
    coords = embed_data(x, manifold.ambient_dim, method=method, seed=seed,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    return manifold.project(coords)


def masked_neighbor_reconstruction(
    values,
    latent_points,
    manifold: Manifold,
    k: int = 8,
    mask_ratio: float = 0.25,
    seed: int | None = None,
) -> ReconstructionResult:
    """Reconstruct masked values from k-nearest latent neighbors.

    Predictions are inverse-distance weighted averages in the manifold's
    ambient coordinates.  This is the baseline metric; a neural model
    (TopoVAE) will replace it in a later iteration.
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
        return ReconstructionResult(
            float("nan"), y.copy(),
            np.ones(n, dtype=bool),
            np.zeros((2, 0), dtype=int),
            np.array([]),
        )

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
