"""Tests for the full manifold catalogue in eso.topology.manifolds."""

import numpy as np
import pytest

from eso.topology.manifolds import get_manifold


ALL_MANIFOLDS = [
    # name,             expected_dim, expected_ambient
    ("circle",          1, 2),
    ("sphere2",         2, 3),
    ("sphere3",         3, 4),
    ("torus2",          2, 3),
    ("torus3",          3, 6),
    ("cylinder",        2, 3),
    ("cone",            2, 3),
    ("klein_bottle",    2, 3),
    ("mobius",          2, 3),
    ("hyperbolic",      2, 2),
    ("h2_r",            3, 3),
    ("plane2d",         2, 2),
    ("plane3d",         3, 3),
    ("plane4d",         4, 4),
    ("s1_r2",           3, 4),
    ("s2_r",            3, 4),
    ("t2_r",            3, 5),
    ("s2_s1",           3, 5),
]


@pytest.mark.parametrize("name,expected_dim,expected_ambient", ALL_MANIFOLDS)
def test_manifold_metadata(name, expected_dim, expected_ambient):
    m = get_manifold(name)
    assert m.dim == expected_dim, f"{name}: dim {m.dim} != {expected_dim}"
    assert m.ambient_dim == expected_ambient, f"{name}: ambient {m.ambient_dim} != {expected_ambient}"


@pytest.mark.parametrize("name,_d,_a", ALL_MANIFOLDS)
def test_sample_uniform_shape(name, _d, _a):
    m = get_manifold(name)
    pts = m.sample_uniform(50, seed=0)
    assert pts.shape == (50, m.ambient_dim), f"{name}: sample shape {pts.shape}"
    assert np.isfinite(pts).all(), f"{name}: non-finite values in sample"


@pytest.mark.parametrize("name,_d,ambient", ALL_MANIFOLDS)
def test_project_from_ambient(name, _d, ambient):
    """project() on ambient-dim data should return same shape and finite values."""
    m = get_manifold(name)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, ambient))
    projected = m.project(x)
    assert projected.shape == (30, ambient), f"{name}: project shape {projected.shape}"
    assert np.isfinite(projected).all(), f"{name}: non-finite in projection"


@pytest.mark.parametrize("name,_d,ambient", ALL_MANIFOLDS)
def test_project_from_low_dim_data(name, _d, ambient):
    """project() should handle input with fewer columns than ambient_dim (padding)."""
    m = get_manifold(name)
    rng = np.random.default_rng(7)
    low_dim = max(1, ambient - 1)
    x = rng.normal(size=(20, low_dim))
    projected = m.project(x)
    assert projected.shape == (20, ambient), f"{name}: low-dim project shape {projected.shape}"
    assert np.isfinite(projected).all(), f"{name}: non-finite in low-dim projection"


@pytest.mark.parametrize("name,_d,_a", ALL_MANIFOLDS)
def test_knn_graph_runs(name, _d, _a):
    m = get_manifold(name)
    pts = m.sample_uniform(40, seed=1)
    edges, weights = m.knn_graph(pts, k=4)
    assert edges.shape[0] == 2
    assert weights.shape[0] == edges.shape[1]
    assert np.isfinite(weights).all()


def test_get_manifold_aliases():
    """Key aliases resolve to the correct class."""
    assert get_manifold("hyperbolic").name == "hyperbolic"
    assert get_manifold("h2").name == "hyperbolic"
    assert get_manifold("poincare").name == "hyperbolic"
    assert get_manifold("mobius_band").name == "mobius"
    assert get_manifold("torus_drift").name == "t2_r"
    assert get_manifold("sphere_line").name == "s2_r"
    assert get_manifold("t3").name == "torus3"


def test_get_manifold_unknown_raises():
    with pytest.raises(ValueError, match="Unknown manifold"):
        get_manifold("unicorn_manifold")


def test_hyperbolic_disk_stays_in_unit_disk():
    m = get_manifold("hyperbolic")
    pts = m.sample_uniform(200, seed=5)
    norms = np.linalg.norm(pts, axis=1)
    assert (norms < 1.0).all(), f"Points outside unit disk: max norm = {norms.max()}"
    # projection also stays inside
    x = np.random.default_rng(5).normal(size=(50, 2)) * 5  # far outside
    proj = m.project(x)
    assert (np.linalg.norm(proj, axis=1) <= 0.971).all()


def test_cone_apex_at_origin():
    """Cone samples should have z ≈ r_xy (cone constraint)."""
    m = get_manifold("cone")
    pts = m.sample_uniform(200, seed=3)
    r_xy = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    diff = np.abs(pts[:, 2] - r_xy)
    assert diff.max() < 1e-10, f"Cone constraint violated: max diff = {diff.max()}"


def test_s1_r2_circle_component_unit_norm():
    m = get_manifold("s1_r2")
    pts = m.sample_uniform(100, seed=2)
    norms = np.linalg.norm(pts[:, :2], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), "S1 component not on unit circle"


def test_s2_s1_both_components_unit_norm():
    m = get_manifold("s2_s1")
    pts = m.sample_uniform(100, seed=9)
    sph_norms = np.linalg.norm(pts[:, :3], axis=1)
    circ_norms = np.linalg.norm(pts[:, 3:5], axis=1)
    assert np.allclose(sph_norms, 1.0, atol=1e-10), "S2 component not on unit sphere"
    assert np.allclose(circ_norms, 1.0, atol=1e-10), "S1 component not on unit circle"
