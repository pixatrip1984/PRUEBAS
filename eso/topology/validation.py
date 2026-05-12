"""Cross-validation helpers for topology tests."""

from __future__ import annotations

import math

import numpy as np

from .evaluator import evaluate_manifold
from .reconstruction import clear_embed_cache


def validate_manifold(
    data,
    manifold: str,
    k: int = 8,
    mask_ratio: float = 0.25,
    n_masks: int = 5,
    seed: int | None = None,
    complexity_weight: float = 1e-8,
    projection_method: str = "linear",
    proj_neighbors: int = 15,
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
                projection_method=projection_method,
                proj_neighbors=proj_neighbors,
            )
        )
    errors = np.array([r["reconstruction_error"] for r in runs], dtype=float)
    scores = np.array([r["score"] for r in runs], dtype=float)
    rtf_vals = np.array([r.get("relative_to_flat", float("nan")) for r in runs], dtype=float)

    base = dict(runs[0])
    base["validation_runs"] = runs
    base["n_masks"] = int(n_masks)
    base["reconstruction_error_mean"] = float(np.nanmean(errors))
    base["reconstruction_error_std"] = float(np.nanstd(errors))
    base["score_mean"] = float(np.nanmean(scores))
    base["score_std"] = float(np.nanstd(scores))
    base["relative_to_flat_mean"] = float(np.nanmean(rtf_vals))
    base["relative_to_flat_std"] = float(np.nanstd(rtf_vals))
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
    projection_method: str = "linear",
    proj_neighbors: int = 15,
) -> list[dict]:
    # Clear the embedding cache between manifold sets so different datasets
    # don't share stale embeddings, but manifolds within the same call reuse them.
    clear_embed_cache()
    results = [
        validate_manifold(
            data,
            manifold=m,
            k=k,
            mask_ratio=mask_ratio,
            n_masks=n_masks,
            seed=None if seed is None else int(seed) + i,
            complexity_weight=complexity_weight,
            projection_method=projection_method,
            proj_neighbors=proj_neighbors,
        )
        for i, m in enumerate(manifolds)
    ]
    return sorted(results, key=lambda r: r["score_mean"] if np.isfinite(r["score_mean"]) else np.inf)
