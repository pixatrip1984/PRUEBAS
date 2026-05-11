"""Candidate manifolds for ESO topology tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class Manifold:
    name: str
    dim: int
    ambient_dim: int
    is_compact: bool = True

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        diff = pts[:, None, :] - pts[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def knn_graph(self, points: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=float)
        k_eff = min(k + 1, len(pts))
        nn = NearestNeighbors(n_neighbors=k_eff).fit(pts)
        dist, idx = nn.kneighbors(pts)
        src = np.repeat(np.arange(len(pts)), k_eff - 1)
        dst = idx[:, 1:].reshape(-1)
        weights = dist[:, 1:].reshape(-1)
        return np.stack([src, dst], axis=0), weights


class Circle(Manifold):
    def __init__(self):
        super().__init__("circle", dim=1, ambient_dim=2, is_compact=True)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 2:
            x = np.pad(x, [(0, 0), (0, 2 - x.shape[-1])])
        y = x[:, :2]
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


class Sphere(Manifold):
    def __init__(self, dim: int = 2):
        super().__init__(f"sphere{dim}", dim=dim, ambient_dim=dim + 1, is_compact=True)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n_points, self.ambient_dim))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < self.ambient_dim:
            x = np.pad(x, [(0, 0), (0, self.ambient_dim - x.shape[-1])])
        y = x[:, : self.ambient_dim]
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


class Torus(Manifold):
    def __init__(self, major_radius: float = 2.0, minor_radius: float = 0.75):
        super().__init__("torus2", dim=2, ambient_dim=3, is_compact=True)
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 2 * np.pi, n_points)
        v = rng.uniform(0, 2 * np.pi, n_points)
        r = self.major_radius + self.minor_radius * np.cos(v)
        return np.stack([r * np.cos(u), r * np.sin(u), self.minor_radius * np.sin(v)], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        y = x[:, :3]
        theta = np.arctan2(y[:, 1], y[:, 0])
        rho = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        phi = np.arctan2(y[:, 2], rho - self.major_radius)
        r = self.major_radius + self.minor_radius * np.cos(phi)
        return np.stack([r * np.cos(theta), r * np.sin(theta), self.minor_radius * np.sin(phi)], axis=1)


class Cylinder(Manifold):
    def __init__(self):
        super().__init__("cylinder", dim=2, ambient_dim=3, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        z = rng.uniform(-1.0, 1.0, n_points)
        return np.stack([np.cos(theta), np.sin(theta), z], axis=1)


class KleinBottle(Manifold):
    def __init__(self):
        super().__init__("klein_bottle", dim=2, ambient_dim=3, is_compact=True)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 2 * np.pi, n_points)
        v = rng.uniform(0, 2 * np.pi, n_points)
        x = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.cos(u)
        y = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.sin(u)
        z = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)
        return np.stack([x, y, z], axis=1)


def get_manifold(name: str) -> Manifold:
    key = name.lower().replace("-", "_")
    if key in {"s1", "circle"}:
        return Circle()
    if key in {"s2", "sphere", "sphere2"}:
        return Sphere(2)
    if key in {"t2", "torus", "torus2"}:
        return Torus()
    if key == "cylinder":
        return Cylinder()
    if key in {"klein", "klein_bottle"}:
        return KleinBottle()
    raise ValueError(f"Unknown manifold: {name}")
