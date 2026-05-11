"""
SHSE Encoder — Python port (fixed geometry, no JS required).

Maps a time-series window (lookback, n_features) to a PyG Data graph.
Each timestep becomes one node on a Fibonacci sphere.
The GNN learns relational structure on top of this fixed geometry.
"""

import numpy as np
import torch
from torch_geometric.data import Data


def fibonacci_sphere(n: int) -> np.ndarray:
    """n equidistant points on unit sphere via Fibonacci lattice."""
    i = np.arange(n, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / max(1, n - 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    a = golden * i
    pts = np.stack([r * np.cos(a), y, r * np.sin(a)], axis=1)
    return pts.astype(np.float32)


def build_knn_graph(points: np.ndarray, k: int):
    """
    Build directed k-NN graph on sphere using angular distance.
    Returns edge_index [2, N*k] and edge_attr [N*k, 1].
    """
    n = len(points)
    cos_sim = np.clip(points @ points.T, -1.0, 1.0)
    angles = np.arccos(cos_sim).astype(np.float32)
    np.fill_diagonal(angles, np.inf)

    knn_idx = np.argsort(angles, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = knn_idx.ravel()
    ang_dist = angles[src, dst]

    edge_index = np.stack([src, dst], axis=0)
    edge_attr = ang_dist[:, None]
    return edge_index.astype(np.int64), edge_attr


class SHSEEncoder:
    """
    Fixed-geometry SHSE encoder.

    The Fibonacci sphere is pre-computed once. Each call to encode()
    maps a window of shape (lookback, n_features) to a PyG Data object
    where node i corresponds to timestep i.

    Node features = [normalized_market_features (9) | sphere_xyz (3)] = 12-dim.
    """

    def __init__(self, lookback: int = 64, n_features: int = 9, neighbors: int = 8):
        self.lookback = lookback
        self.n_features = n_features
        self.neighbors = neighbors
        self.input_dim = n_features + 3  # features + sphere position

        pts = fibonacci_sphere(lookback)
        ei, ea = build_knn_graph(pts, neighbors)

        self._edge_index = torch.from_numpy(ei)
        self._edge_attr = torch.from_numpy(ea)
        self._positions = torch.from_numpy(pts)

    def encode(self, window: np.ndarray, label: int) -> Data:
        """
        window : (lookback, n_features) float32
        label  : 0 (price falls) or 1 (price rises)
        """
        # Robust per-feature normalization using IQR
        q1 = np.percentile(window, 25, axis=0)
        q3 = np.percentile(window, 75, axis=0)
        scale = np.maximum(q3 - q1, 1e-6)
        median = np.median(window, axis=0)
        normed = np.clip((window - median) / scale, -4.0, 4.0).astype(np.float32)

        # Concatenate market features with sphere position
        x = np.concatenate([normed, self._positions.numpy()], axis=1)  # (lookback, 12)

        return Data(
            x=torch.from_numpy(x),
            edge_index=self._edge_index.clone(),
            edge_attr=self._edge_attr.clone(),
            y=torch.tensor([label], dtype=torch.long),
        )
