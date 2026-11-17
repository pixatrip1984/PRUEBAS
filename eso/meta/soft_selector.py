"""Non-discarding, family-prioritized selection and method crossover generation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Any

from eso.db.lab_store import DEFAULT_ARSENAL, RelationalLabStore
from eso.meta.signature import diagnosis_signature


_SPEC_BY_ID = {str(spec["method_id"]): dict(spec) for spec in DEFAULT_ARSENAL}
_SPEC_BY_NAME = {str(spec["name"]).lower(): dict(spec) for spec in DEFAULT_ARSENAL}

_FAMILY_COMPLEMENTS = {
    frozenset(("periodic_compact", "curved_hyperbolic")),
    frozenset(("spectral_causal", "cascade_branching")),
    frozenset(("microstructure_flow", "topological_invariant")),
    frozenset(("flat_euclidean", "curved_spherical")),
    frozenset(("non_compact_product", "topological_invariant")),
    frozenset(("curved_spherical", "cascade_branching")),
    frozenset(("singular_transition", "spectral_causal")),
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _features(diagnosis: Mapping[str, Any]) -> dict[str, float]:
    if any(key in diagnosis for key in ("dimension", "stationarity", "lyapunov", "symmetries")):
        return diagnosis_signature(dict(diagnosis))
    return {
        "dim_consensus": _finite(diagnosis.get("dim_consensus"), float("nan")),
        "chaotic_hint": _finite(diagnosis.get("chaotic_hint")),
        "stationary_hint": _finite(diagnosis.get("stationary_hint")),
        "periodic_hint": _finite(diagnosis.get("periodic_hint")),
        "dominant_frequency": _finite(diagnosis.get("dominant_frequency"), float("nan")),
        "lyapunov_max": _finite(diagnosis.get("lyapunov_max"), float("nan")),
    }


def _candidate_spec(candidate: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        item = dict(candidate)
        method_id = str(item.get("method_id") or item.get("name") or "")
        if not method_id:
            raise ValueError("candidate mappings require method_id or name")
        known = _SPEC_BY_ID.get(method_id) or _SPEC_BY_NAME.get(str(item.get("name", "")).lower()) or {}
        return {
            **known,
            **item,
            "method_id": method_id,
            "name": str(item.get("name") or known.get("name") or method_id),
            "family": str(item.get("family") or known.get("family") or "unclassified"),
            "algebraic_group": str(item.get("algebraic_group") or known.get("algebraic_group") or "unknown"),
        }
    key = str(candidate)
    known = _SPEC_BY_ID.get(key) or _SPEC_BY_NAME.get(key.lower())
    if known:
        return dict(known)
    return {"method_id": key, "name": key, "family": "unclassified", "algebraic_group": "unknown", "category_tags": []}


class FamilyPrioritizedSelector:
    """Rank mathematical methods while retaining a positive probability for all."""

    def __init__(self, store: RelationalLabStore | None = None, epsilon: float = 0.01) -> None:
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must satisfy 0 < epsilon < 1")
        self.store = store
        self.epsilon = float(epsilon)

    def _candidates(self, candidates: Iterable[str | Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        if candidates is not None:
            items = [_candidate_spec(candidate) for candidate in candidates]
        elif self.store is not None:
            items = self.store.list_methods()
            if not items:
                items = [dict(spec) for spec in DEFAULT_ARSENAL]
        else:
            items = [dict(spec) for spec in DEFAULT_ARSENAL]
        seen: set[str] = set()
        unique = []
        for item in items:
            method_id = str(item["method_id"])
            if method_id not in seen:
                seen.add(method_id)
                unique.append(item)
        return unique

    def _history(self) -> dict[str, dict[str, Any]]:
        if self.store is None:
            return {}
        return {str(row["family"]): row for row in self.store.query_family_performance()}

    def _family_score(self, family: str, features: Mapping[str, float]) -> float:
        score = 0.0
        dim = _finite(features.get("dim_consensus"), float("nan"))
        periodic = _finite(features.get("periodic_hint")) > 0.0
        chaotic = _finite(features.get("chaotic_hint")) > 0.0
        stationary = _finite(features.get("stationary_hint")) > 0.0

        if periodic and family == "periodic_compact":
            score += 2.4
        if periodic and family in {"non_compact_product", "spectral_causal"}:
            score += 0.9
        if chaotic and family == "curved_hyperbolic":
            score += 2.0
        if chaotic and family in {"cascade_branching", "non_compact_product"}:
            score += 1.1
        if stationary and family in {"periodic_compact", "curved_spherical", "topological_invariant"}:
            score += 0.6
        if not stationary and family in {"non_compact_product", "flat_euclidean", "singular_transition"}:
            score += 0.35

        if math.isfinite(dim):
            if dim <= 1.4 and family == "periodic_compact":
                score += 1.4
            elif 1.4 < dim <= 2.6 and family in {"curved_hyperbolic", "curved_spherical", "topological_invariant"}:
                score += 1.2
            elif 2.6 < dim <= 3.6 and family in {"non_compact_product", "curved_spherical", "periodic_compact"}:
                score += 1.0
            elif dim > 3.6 and family in {"flat_euclidean", "spectral_causal"}:
                score += 0.8

        return score

    @staticmethod
    def _historical_score(performance: Mapping[str, Any] | None) -> float:
        if not performance:
            return 0.0
        relative = _finite(performance.get("mean_relative_to_flat"))
        stability = _finite(performance.get("mean_oos_stability"))
        metric = _finite(performance.get("mean_metric_score"))
        count = max(0.0, _finite(performance.get("experiment_count")))
        confidence = 1.0 - math.exp(-count / 3.0)
        signal = 0.7 * math.tanh(relative) + 0.5 * math.tanh(stability) + 0.3 * math.tanh(metric)
        return confidence * signal

    def score_candidates(
        self,
        diagnosis: Mapping[str, Any],
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        items = self._candidates(candidates)
        if not items:
            return []
        if self.epsilon * len(items) > 1.0 + 1e-12:
            raise ValueError("epsilon is too large for the number of candidates")

        features = _features(diagnosis)
        history = self._history()
        scored: list[dict[str, Any]] = []
        for order, item in enumerate(items):
            family = str(item.get("family", "unclassified"))
            structural_score = self._family_score(family, features)
            historical_score = self._historical_score(history.get(family))
            score = structural_score + historical_score
            scored.append(
                {
                    "method_id": str(item["method_id"]),
                    "name": str(item.get("name", item["method_id"])),
                    "family": family,
                    "algebraic_group": str(item.get("algebraic_group", "unknown")),
                    "structural_score": structural_score,
                    "historical_score": historical_score,
                    "score": score,
                    "_order": order,
                }
            )

        maximum = max(row["score"] for row in scored)
        exp_scores = [math.exp(row["score"] - maximum) for row in scored]
        total = sum(exp_scores)
        residual = max(0.0, 1.0 - self.epsilon * len(scored))
        for row, exp_score in zip(scored, exp_scores):
            probability = self.epsilon + residual * (exp_score / total)
            row["probability"] = probability
            row["weight"] = probability

        scored.sort(key=lambda row: (-row["score"], row["_order"]))
        for rank, row in enumerate(scored, start=1):
            row["rank"] = rank
            row.pop("_order", None)
        return scored

    def prioritize(
        self,
        diagnosis: Mapping[str, Any],
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self.score_candidates(diagnosis, candidates)

    def rank_candidates(
        self,
        diagnosis: Mapping[str, Any],
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self.score_candidates(diagnosis, candidates)

    def recommend(
        self,
        diagnosis: Mapping[str, Any],
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self.score_candidates(diagnosis, candidates)

    def selection_weights(
        self,
        diagnosis: Mapping[str, Any],
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        return {row["method_id"]: float(row["probability"]) for row in self.score_candidates(diagnosis, candidates)}


class MethodCrossoverGenerator:
    """Generate deterministic cross-family hypotheses without dropping families."""

    def __init__(self, store: RelationalLabStore | None = None) -> None:
        self.store = store

    def _candidates(self, candidates: Iterable[str | Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        if candidates is not None:
            return [_candidate_spec(candidate) for candidate in candidates]
        if self.store is not None:
            methods = self.store.list_methods()
            if methods:
                return methods
        return [dict(spec) for spec in DEFAULT_ARSENAL]

    @staticmethod
    def _is_complementary(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        family_a = str(a.get("family", "unclassified"))
        family_b = str(b.get("family", "unclassified"))
        if family_a == family_b:
            return False
        pair = frozenset((family_a, family_b))
        if pair in _FAMILY_COMPLEMENTS:
            return True
        return str(a.get("algebraic_group")) != str(b.get("algebraic_group"))

    @staticmethod
    def _proposal(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        method_a = str(a["method_id"])
        method_b = str(b["method_id"])
        family_a = str(a.get("family", "unclassified"))
        family_b = str(b.get("family", "unclassified"))
        digest = hashlib.sha256(f"{method_a}|{method_b}".encode("utf-8")).hexdigest()[:16]
        return {
            "crossover_id": f"xover-{digest}",
            "parent_method_a": method_a,
            "parent_method_b": method_b,
            "emergent_family": f"{family_a}_x_{family_b}",
            "hypothesis": (
                f"{method_a} x {method_b} should combine {family_a} and {family_b}; "
                "test whether relative_to_flat improves while out-of-sample stability "
                "does not fall below the stronger parent."
            ),
            "metrics": {},
            "status": "proposed",
        }

    def propose_crossovers(
        self,
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
        *,
        limit: int | None = None,
        persist: bool = False,
    ) -> list[dict[str, Any]]:
        items = self._candidates(candidates)
        proposals: list[dict[str, Any]] = []
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if not self._is_complementary(a, b):
                    continue
                proposals.append(self._proposal(a, b))
                if limit is not None and len(proposals) >= limit:
                    break
            if limit is not None and len(proposals) >= limit:
                break

        if persist:
            if self.store is None:
                raise ValueError("persist=True requires a RelationalLabStore")
            self.persist_proposals(proposals)
        return proposals

    def generate(
        self,
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
        *,
        limit: int | None = None,
        persist: bool = False,
    ) -> list[dict[str, Any]]:
        return self.propose_crossovers(candidates, limit=limit, persist=persist)

    def propose(
        self,
        candidates: Iterable[str | Mapping[str, Any]] | None = None,
        *,
        limit: int | None = None,
        persist: bool = False,
    ) -> list[dict[str, Any]]:
        return self.propose_crossovers(candidates, limit=limit, persist=persist)

    def persist_proposals(self, proposals: Iterable[Mapping[str, Any]]) -> list[str]:
        if self.store is None:
            raise ValueError("a RelationalLabStore is required to persist crossovers")
        ids = []
        for proposal in proposals:
            ids.append(
                self.store.record_crossover(
                    str(proposal["parent_method_a"]),
                    str(proposal["parent_method_b"]),
                    str(proposal["emergent_family"]),
                    str(proposal["hypothesis"]),
                    metrics=proposal.get("metrics", {}),
                    status=str(proposal.get("status", "proposed")),
                    crossover_id=str(proposal["crossover_id"]),
                )
            )
        return ids

    def query_crossovers(self, *, status: str | None = None, emergent_family: str | None = None) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        return self.store.query_crossovers(status=status, emergent_family=emergent_family)
