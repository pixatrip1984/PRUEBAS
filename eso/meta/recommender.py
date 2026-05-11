"""Manifold recommenders for ESO."""

from __future__ import annotations

import numpy as np


class HeuristicRecommender:
    """Cold-start manifold recommender based on blind-diagnosis hints."""

    def recommend(self, diagnosis: dict, candidates: list[str] | None = None) -> list[str]:
        candidates = candidates or ["circle", "sphere2", "torus2", "cylinder"]
        dim = diagnosis.get("dimension", {}).get("consensus_dimension", np.nan)
        periodic = bool(diagnosis.get("symmetries", {}).get("periodic_hint", False))
        chaotic = bool(diagnosis.get("lyapunov", {}).get("chaotic_hint", False))
        stationary = diagnosis.get("stationarity", {}).get("label") == "stationary"

        scores = {name: 0.0 for name in candidates}
        for name in candidates:
            key = name.lower()
            if periodic and key in {"circle", "torus2"}:
                scores[name] += 2.0
            if chaotic and key in {"sphere2", "torus2", "cylinder"}:
                scores[name] += 1.0
            if stationary and key in {"circle", "torus2"}:
                scores[name] += 0.5
            if np.isfinite(dim):
                if dim <= 1.4 and key == "circle":
                    scores[name] += 2.0
                elif 1.4 < dim <= 2.6 and key in {"sphere2", "torus2"}:
                    scores[name] += 2.0
                elif dim > 2.6 and key in {"torus2", "cylinder"}:
                    scores[name] += 1.0
        return sorted(candidates, key=lambda n: (-scores[n], candidates.index(n)))
