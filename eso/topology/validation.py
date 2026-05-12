"""Cross-validation helpers for topology tests."""

from __future__ import annotations

import math

import numpy as np

from .evaluator import evaluate_manifold


def validate_manifold(
    data,
    manifold: str,
    k: int = 8,
    mask_ratio: float = 0.25,
    n_masks: int = 5,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
) -> dict:
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(int(n_masks)):
        run_seed = int(rng.integers(0, 2**31 - 1))
        runs.append(
            evaluate_manifold(
                data,
                manifold,
                k=k,
                mask_ratio=mask_ratio,
                seed=run_seed,
                complexity_weight=complexity_weight,
            )
        )
    errors = np.array([r["reconstruction_error"] for r in runs], dtype=float)
    scores = np.array([r["score"] for r in runs], dtype=float)
    base = dict(runs[0])
    base["validation_runs"] = runs
    base["n_masks"] = int(n_masks)
    base["reconstruction_error_mean"] = float(np.nanmean(errors))
    base["reconstruction_error_std"] = float(np.nanstd(errors))
    base["score_mean"] = float(np.nanmean(scores))
    base["score_std"] = float(np.nanstd(scores))
    if len(errors) > 1:
        base["reconstruction_error_ci95"] = float(1.96 * np.nanstd(errors) / math.sqrt(len(errors)))
    else:
        base["reconstruction_error_ci95"] = 0.0
    return base


def validate_manifolds(
    data,
    manifolds: list[str],
    k: int = 8,
    mask_ratio: float = 0.25,
    n_masks: int = 5,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
) -> list[dict]:
    results = [
        validate_manifold(
            data,
            manifold=m,
            k=k,
            mask_ratio=mask_ratio,
            n_masks=n_masks,
            seed=None if seed is None else int(seed) + i,
            complexity_weight=complexity_weight,
        )
        for i, m in enumerate(manifolds)
    ]
    return sorted(results, key=lambda r: r["score_mean"] if np.isfinite(r["score_mean"]) else np.inf)
