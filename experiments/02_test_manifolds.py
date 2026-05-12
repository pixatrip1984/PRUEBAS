"""Compare candidate manifolds with ESO's masked reconstruction metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eso.pipeline import ESOExplorer


def harmonic(n: int = 1000):
    t = np.linspace(0.0, 8.0 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def torus(n: int = 1000):
    t = np.linspace(0.0, 12.0 * np.pi, n)
    r = np.sqrt(2.0)
    return np.stack([np.cos(t), np.sin(t), np.cos(r * t), np.sin(r * t)], axis=1)


def butterfly(n: int = 1000):
    t = np.linspace(0.0, 24.0 * np.pi, n)
    radius = np.sin(2.0 * t) * np.cos(0.5 * t)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.sin(0.25 * t)], axis=1)


def dataset(name: str, n: int):
    key = name.lower()
    if key == "harmonic":
        return harmonic(n)
    if key == "torus":
        return torus(n)
    if key in {"butterfly", "lorenz_proxy"}:
        return butterfly(n)
    raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="harmonic")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--manifolds", nargs="+", default=["circle", "sphere2", "torus2"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    args = parser.parse_args()

    data = dataset(args.dataset, args.n)
    explorer = ESOExplorer(registry_path="experiments/eso_registry.csv")
    report = explorer.explore(
        data,
        dataset_id=args.dataset,
        manifolds=args.manifolds,
        k=args.k,
        mask_ratio=args.mask_ratio,
        seed=123,
        save=False,
    )
    print("best:", report["best"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
