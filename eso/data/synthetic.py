import numpy as np


def harmonic(n=1000, seed=None):
    t = np.linspace(0.0, 8.0 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def torus(n=1000, seed=None):
    t = np.linspace(0.0, 12.0 * np.pi, n)
    ratio = np.sqrt(2.0)
    return np.stack([np.cos(t), np.sin(t), np.cos(ratio * t), np.sin(ratio * t)], axis=1)


def butterfly(n=1000, seed=None):
    t = np.linspace(0.0, 24.0 * np.pi, n)
    r = np.sin(2.0 * t) * np.cos(0.5 * t)
    return np.stack([r * np.cos(t), r * np.sin(t), np.sin(0.25 * t)], axis=1)


def generate_dataset(name, n=1000, seed=None):
    key = str(name).lower()
    if key in {"harmonic", "circle", "s1"}:
        return harmonic(n=n, seed=seed)
    if key in {"torus", "t2"}:
        return torus(n=n, seed=seed)
    if key in {"butterfly", "lorenz_proxy"}:
        return butterfly(n=n, seed=seed)
    raise ValueError(f"unknown dataset: {name}")
