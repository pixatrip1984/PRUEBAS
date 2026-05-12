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

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        y = x[:, :3]
        circ = y[:, :2] / np.maximum(np.linalg.norm(y[:, :2], axis=1, keepdims=True), 1e-12)
        return np.concatenate([circ, y[:, 2:3]], axis=1)


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

    def project(self, x: np.ndarray) -> np.ndarray:
        # NOTE: Proper projection to the nearest point on the Klein bottle
        # requires a 2D non-linear optimisation per point (expensive).
        # This is a first-order approximation: treat the Klein bottle as an
        # immersed surface and snap to the nearest sample on a dense grid.
        # TODO: replace with iterative nearest-point solver for rigour.
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        p = x[:, :3]
        # Generate a coarse reference grid on the Klein bottle surface
        rng = np.random.default_rng(0)
        u = rng.uniform(0, 2 * np.pi, 4000)
        v = rng.uniform(0, 2 * np.pi, 4000)
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_ht, sin_ht = np.cos(u / 2), np.sin(u / 2)
        gx = (2 + cos_ht * np.sin(v) - sin_ht * np.sin(2 * v)) * cos_u
        gy = (2 + cos_ht * np.sin(v) - sin_ht * np.sin(2 * v)) * sin_u
        gz = sin_ht * np.sin(v) + cos_ht * np.sin(2 * v)
        grid = np.stack([gx, gy, gz], axis=1)
        # Nearest-grid-point projection (approximate)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(grid)
        _, idx = nn.kneighbors(p)
        return grid[idx[:, 0]]


# ── Flat baselines ────────────────────────────────────────────────────────────

class Plane(Manifold):
    """ℝⁿ — flat Euclidean baseline, no curvature assumption."""

    def __init__(self, ambient_dim: int = 2):
        super().__init__(f"plane{ambient_dim}d", dim=ambient_dim, ambient_dim=ambient_dim, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, (n_points, self.ambient_dim))

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] >= self.ambient_dim:
            return x[:, : self.ambient_dim]
        return np.pad(x, [(0, 0), (0, self.ambient_dim - x.shape[-1])])


# ── Hyperbolic spaces ─────────────────────────────────────────────────────────

class HyperbolicDisk(Manifold):
    """H² — Poincaré disk model (open unit disk).
    Captures exponential distance growth, heavy-tail / power-law structure.
    Points near the boundary are 'infinitely far' from the centre in the
    geodesic metric — relevant for scale-free financial dynamics.
    """

    def __init__(self):
        super().__init__("hyperbolic", dim=2, ambient_dim=2, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Uniform in the Euclidean disk (approximation — geodesically uniform
        # sampling would weight toward the boundary, but this suffices for
        # the k-NN relational reconstruction baseline).
        r = np.sqrt(rng.uniform(0.0, 1.0, n_points)) * 0.97
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 2:
            x = np.pad(x, [(0, 0), (0, 2 - x.shape[-1])])
        y = x[:, :2]
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        # Clamp inside the open unit disk with max radius 0.97
        scale = np.minimum(1.0, 0.97 / np.maximum(norms, 1e-8))
        return y * scale


class HyperbolicCylinder(Manifold):
    """H² × ℝ — hyperbolic disk plus linear drift.
    Combines scale-free oscillation with persistent trend.
    """

    def __init__(self):
        super().__init__("h2_r", dim=3, ambient_dim=3, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r = np.sqrt(rng.uniform(0.0, 1.0, n_points)) * 0.97
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        z = rng.uniform(-1.0, 1.0, n_points)
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        p = x[:, :3]
        xy = p[:, :2]
        norms = np.linalg.norm(xy, axis=1, keepdims=True)
        scale = np.minimum(1.0, 0.97 / np.maximum(norms, 1e-8))
        return np.concatenate([xy * scale, p[:, 2:3]], axis=1)


# ── Non-orientable surfaces ───────────────────────────────────────────────────

class MobiusBand(Manifold):
    """Möbius band — non-compact non-orientable 2-manifold.
    The single half-twist means going around the band once flips orientation;
    could capture mean-reversal symmetry breaking in BTC returns.
    """

    def __init__(self):
        super().__init__("mobius", dim=2, ambient_dim=3, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = rng.uniform(0.0, 2.0 * np.pi, n_points)
        s = rng.uniform(-0.5, 0.5, n_points)
        cos_t, sin_t = np.cos(t), np.sin(t)
        cos_ht, sin_ht = np.cos(t / 2), np.sin(t / 2)
        x = (1.0 + s * cos_ht) * cos_t
        y = (1.0 + s * cos_ht) * sin_t
        z = s * sin_ht
        return np.stack([x, y, z], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        p = x[:, :3]
        t = np.arctan2(p[:, 1], p[:, 0]) % (2.0 * np.pi)
        r_xy = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        cos_ht, sin_ht = np.cos(t / 2), np.sin(t / 2)
        # Estimate s from two equations, average and clamp
        eps = 1e-8
        s1 = np.where(np.abs(cos_ht) > 0.1, (r_xy - 1.0) / (cos_ht + eps), 0.0)
        s2 = np.where(np.abs(sin_ht) > 0.1, p[:, 2] / (sin_ht + eps), 0.0)
        s = np.clip(0.5 * (s1 + s2), -0.5, 0.5)
        out_x = (1.0 + s * cos_ht) * np.cos(t)
        out_y = (1.0 + s * cos_ht) * np.sin(t)
        out_z = s * sin_ht
        return np.stack([out_x, out_y, out_z], axis=1)


# ── Degenerate / singular geometries ─────────────────────────────────────────

class Cone(Manifold):
    """Half-cone z = r_xy — degenerate metric at the apex.
    Models volatility regime transitions that converge toward a single state
    (the tip) and diverge outward.  ambient ℝ³.
    """

    def __init__(self):
        super().__init__("cone", dim=2, ambient_dim=3, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r = rng.uniform(0.0, 1.0, n_points)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        return np.stack([r * np.cos(theta), r * np.sin(theta), r], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 3:
            x = np.pad(x, [(0, 0), (0, 3 - x.shape[-1])])
        p = x[:, :3]
        theta = np.arctan2(p[:, 1], p[:, 0])
        # Cone surface: z = r_xy, so take r = (r_xy + max(z,0)) / 2 as estimate
        r_xy = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        r = np.maximum((r_xy + np.maximum(p[:, 2], 0.0)) * 0.5, 0.0)
        return np.stack([r * np.cos(theta), r * np.sin(theta), r], axis=1)


# ── Product manifolds ─────────────────────────────────────────────────────────

class ProductCylinderLine(Manifold):
    """S¹ × ℝ² — 1-cycle plus 2D drift.
    Models oscillatory BTC patterns (intraday cycle) over a 2D trend plane.
    """

    def __init__(self):
        super().__init__("s1_r2", dim=3, ambient_dim=4, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        drift = rng.uniform(-1.0, 1.0, (n_points, 2))
        return np.concatenate([np.stack([np.cos(theta), np.sin(theta)], axis=1), drift], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 4:
            x = np.pad(x, [(0, 0), (0, 4 - x.shape[-1])])
        circle = x[:, :2] / np.maximum(np.linalg.norm(x[:, :2], axis=1, keepdims=True), 1e-12)
        return np.concatenate([circle, x[:, 2:4]], axis=1)


class SphereLine(Manifold):
    """S² × ℝ — 2-sphere plus linear drift.
    Models BTC as moving on a spherical 'state shell' while trending.
    Distinguishes from S¹×ℝ²: the cyclic component has 2 degrees of freedom.
    """

    def __init__(self):
        super().__init__("s2_r", dim=3, ambient_dim=4, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sph = rng.normal(size=(n_points, 3))
        sph /= np.maximum(np.linalg.norm(sph, axis=1, keepdims=True), 1e-12)
        w = rng.uniform(-1.0, 1.0, (n_points, 1))
        return np.concatenate([sph, w], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 4:
            x = np.pad(x, [(0, 0), (0, 4 - x.shape[-1])])
        sph = x[:, :3] / np.maximum(np.linalg.norm(x[:, :3], axis=1, keepdims=True), 1e-12)
        return np.concatenate([sph, x[:, 3:4]], axis=1)


class TorusDrift(Manifold):
    """T² × ℝ — 2-torus plus linear drift.
    Two independent periodicities (e.g., daily + weekly BTC cycle) plus trend.
    ambient ℝ⁵: (cos θ₁, sin θ₁, cos θ₂, sin θ₂, z).
    """

    def __init__(self):
        super().__init__("t2_r", dim=3, ambient_dim=5, is_compact=False)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t1 = rng.uniform(0.0, 2.0 * np.pi, n_points)
        t2 = rng.uniform(0.0, 2.0 * np.pi, n_points)
        z = rng.uniform(-1.0, 1.0, n_points)
        return np.stack([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), z], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 5:
            x = np.pad(x, [(0, 0), (0, 5 - x.shape[-1])])
        c1 = x[:, :2] / np.maximum(np.linalg.norm(x[:, :2], axis=1, keepdims=True), 1e-12)
        c2 = x[:, 2:4] / np.maximum(np.linalg.norm(x[:, 2:4], axis=1, keepdims=True), 1e-12)
        return np.concatenate([c1, c2, x[:, 4:5]], axis=1)


class Torus3(Manifold):
    """T³ — three-torus: three independent periodic coordinates.
    Tests whether BTC compact features encode three decoupled oscillatory modes.
    ambient ℝ⁶: (cos θ₁, sin θ₁, cos θ₂, sin θ₂, cos θ₃, sin θ₃).
    """

    def __init__(self):
        super().__init__("torus3", dim=3, ambient_dim=6, is_compact=True)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0.0, 2.0 * np.pi, (n_points, 3))
        cols = [np.cos(angles[:, i]) for i in range(3)] + [np.sin(angles[:, i]) for i in range(3)]
        return np.stack([np.cos(angles[:, 0]), np.sin(angles[:, 0]),
                         np.cos(angles[:, 1]), np.sin(angles[:, 1]),
                         np.cos(angles[:, 2]), np.sin(angles[:, 2])], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 6:
            x = np.pad(x, [(0, 0), (0, 6 - x.shape[-1])])
        parts = []
        for i in range(3):
            pair = x[:, 2 * i : 2 * i + 2]
            parts.append(pair / np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-12))
        return np.concatenate(parts, axis=1)


class ProductS2S1(Manifold):
    """S² × S¹ — 2-sphere times circle.
    Spherical state space (e.g., direction of price force) plus periodic modulation.
    ambient ℝ⁵: (x,y,z) on S² + (cos θ, sin θ) on S¹.
    """

    def __init__(self):
        super().__init__("s2_s1", dim=3, ambient_dim=5, is_compact=True)

    def sample_uniform(self, n_points: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sph = rng.normal(size=(n_points, 3))
        sph /= np.maximum(np.linalg.norm(sph, axis=1, keepdims=True), 1e-12)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        circ = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return np.concatenate([sph, circ], axis=1)

    def project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] < 5:
            x = np.pad(x, [(0, 0), (0, 5 - x.shape[-1])])
        sph = x[:, :3] / np.maximum(np.linalg.norm(x[:, :3], axis=1, keepdims=True), 1e-12)
        circ = x[:, 3:5] / np.maximum(np.linalg.norm(x[:, 3:5], axis=1, keepdims=True), 1e-12)
        return np.concatenate([sph, circ], axis=1)


# ── Registry ──────────────────────────────────────────────────────────────────

_MANIFOLD_REGISTRY: dict[str, type] = {}


def _reg(*keys):
    """Decorator: register a factory function under multiple name aliases."""
    def decorator(fn):
        for k in keys:
            _MANIFOLD_REGISTRY[k] = fn
        return fn
    return decorator


def get_manifold(name: str) -> Manifold:
    key = name.lower().replace("-", "_")

    # ── 1-D ──────────────────────────────────────────────────────────────────
    if key in {"s1", "circle"}:
        return Circle()

    # ── 2-D compact orientable ────────────────────────────────────────────────
    if key in {"s2", "sphere", "sphere2"}:
        return Sphere(2)
    if key in {"s3", "sphere3"}:
        return Sphere(3)
    if key in {"s4", "sphere4"}:
        return Sphere(4)
    if key in {"t2", "torus", "torus2"}:
        return Torus()

    # ── 2-D compact non-orientable ────────────────────────────────────────────
    if key in {"klein", "klein_bottle"}:
        return KleinBottle()
    if key in {"mobius", "mobius_band"}:
        return MobiusBand()

    # ── 2-D non-compact ───────────────────────────────────────────────────────
    if key == "cylinder":
        return Cylinder()
    if key in {"cone"}:
        return Cone()
    if key in {"hyperbolic", "h2", "poincare", "hyperbolic_disk"}:
        return HyperbolicDisk()
    if key in {"plane", "plane2d"}:
        return Plane(2)
    if key in {"plane3d"}:
        return Plane(3)
    if key in {"plane4d"}:
        return Plane(4)

    # ── 3-D: hyperbolic products ──────────────────────────────────────────────
    if key in {"h2_r", "hyperbolic_cylinder", "h2_line"}:
        return HyperbolicCylinder()

    # ── 3-D: sphere products ──────────────────────────────────────────────────
    if key in {"s1_r2", "product_cylinder_line"}:
        return ProductCylinderLine()
    if key in {"s2_r", "sphere_line"}:
        return SphereLine()
    if key in {"s2_s1", "product_s2_s1"}:
        return ProductS2S1()

    # ── 3-D: torus products ───────────────────────────────────────────────────
    if key in {"t2_r", "torus_drift", "torus_line"}:
        return TorusDrift()
    if key in {"torus3", "t3"}:
        return Torus3()

    raise ValueError(
        f"Unknown manifold: '{name}'. Available: circle, sphere2, sphere3, sphere4, "
        "torus2, torus3, cylinder, cone, klein_bottle, mobius, hyperbolic, h2_r, "
        "plane2d, plane3d, plane4d, s1_r2, s2_r, t2_r, s2_s1"
    )
