"""Manifold recommenders for ESO."""

from __future__ import annotations

import numpy as np


# Semantic tags for each manifold — used by the heuristic scorer.
# Each entry: (name, tags)
# Tags: "compact", "periodic", "non_compact", "curved_neg", "curved_pos",
#       "product", "non_orientable", "degenerate", "dim_1", "dim_2", "dim_3"
_MANIFOLD_TAGS: dict[str, set[str]] = {
    "circle":      {"compact", "periodic", "dim_1"},
    "sphere2":     {"compact", "curved_pos", "dim_2"},
    "sphere3":     {"compact", "curved_pos", "dim_3"},
    "sphere4":     {"compact", "curved_pos"},
    "torus2":      {"compact", "periodic", "dim_2"},
    "torus3":      {"compact", "periodic", "dim_3"},
    "cylinder":    {"non_compact", "periodic", "dim_2"},
    "cone":        {"non_compact", "degenerate", "dim_2"},
    "klein_bottle":{"compact", "non_orientable", "dim_2"},
    "mobius":      {"non_compact", "non_orientable", "dim_2"},
    "hyperbolic":  {"non_compact", "curved_neg", "dim_2"},
    "h2_r":        {"non_compact", "curved_neg", "product", "dim_3"},
    "plane2d":     {"non_compact", "dim_2"},
    "plane3d":     {"non_compact", "dim_3"},
    "plane4d":     {"non_compact"},
    "s1_r2":       {"non_compact", "periodic", "product", "dim_3"},
    "s2_r":        {"non_compact", "curved_pos", "product", "dim_3"},
    "t2_r":        {"non_compact", "periodic", "product", "dim_3"},
    "s2_s1":       {"compact", "periodic", "curved_pos", "product", "dim_3"},
}


class HeuristicRecommender:
    """Cold-start manifold recommender based on blind-diagnosis hints.

    Scores each candidate based on:
    - Intrinsic dimension estimate
    - Stationarity (compact vs open spaces)
    - Periodicity (torus/circle family)
    - Chaos / Lyapunov (non-compact, curved spaces)
    - Non-stationarity (open/drift spaces preferred)
    """

    def recommend(self, diagnosis: dict, candidates: list[str] | None = None) -> list[str]:
        default_full = [
            "circle", "sphere2", "torus2", "cylinder", "s1_r2",
            "hyperbolic", "cone", "s2_r", "t2_r", "mobius",
        ]
        candidates = candidates or default_full
        dim = diagnosis.get("dimension", {}).get("consensus_dimension", np.nan)
        periodic = bool(diagnosis.get("symmetries", {}).get("periodic_hint", False))
        chaotic = bool(diagnosis.get("lyapunov", {}).get("chaotic_hint", False))
        stationary = diagnosis.get("stationarity", {}).get("label") == "stationary"
        non_stationary = diagnosis.get("stationarity", {}).get("label") == "trend_or_unit_root"

        scores = {name: 0.0 for name in candidates}
        for name in candidates:
            key = name.lower().replace("-", "_")
            tags = _MANIFOLD_TAGS.get(key, set())

            # ── Periodicity hint ───────────────────────────────────────────
            if periodic and "periodic" in tags:
                scores[name] += 2.0

            # ── Stationarity: prefer compact or periodic when stationary ──
            if stationary and ("compact" in tags or "periodic" in tags):
                scores[name] += 0.5

            # ── Non-stationarity: prefer open/drift/degenerate spaces ─────
            if non_stationary and "non_compact" in tags:
                scores[name] += 1.0
            if non_stationary and "degenerate" in tags:
                scores[name] += 0.5

            # ── Chaos: non-compact curved and product spaces ───────────────
            if chaotic:
                if "curved_neg" in tags:          # hyperbolic — chaotic orbits
                    scores[name] += 1.5
                if "curved_pos" in tags:           # spherical — bounded chaos
                    scores[name] += 0.5
                if "product" in tags:
                    scores[name] += 0.5

            # ── Intrinsic dimension alignment ─────────────────────────────
            if np.isfinite(dim):
                if dim <= 1.4 and "dim_1" in tags:
                    scores[name] += 2.0
                elif 1.4 < dim <= 2.6 and "dim_2" in tags:
                    scores[name] += 2.0
                elif 2.6 < dim <= 3.5 and "dim_3" in tags:
                    scores[name] += 2.0
                elif dim > 3.5 and "non_compact" in tags:
                    scores[name] += 1.0

            # ── Non-orientable surfaces: penalise unless dim is ambiguous ─
            if "non_orientable" in tags and not np.isfinite(dim):
                scores[name] += 0.1  # keep in play but don't prioritise

        return sorted(candidates, key=lambda n: (-scores[n], candidates.index(n)))
