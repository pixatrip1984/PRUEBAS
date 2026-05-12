"""Main ESO orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from eso.diagnostics.report import run_diagnosis
from eso.meta.recommender import HeuristicRecommender
from eso.meta.registry import ExperimentRegistry
from eso.meta.signature import experiment_signature
from eso.topology.evaluator import evaluate_manifold, rank_manifolds


@dataclass
class ESOExplorer:
    """Diagnose data, test candidate manifolds, and register experiments."""

    registry_path: str = "experiments/eso_registry.csv"
    default_manifolds: list[str] = field(default_factory=lambda: ["circle", "sphere2", "torus2", "cylinder"])

    def __post_init__(self):
        self.registry = ExperimentRegistry(self.registry_path)
        self.recommender = HeuristicRecommender()
        self._last_report = None

    def diagnose(self, data) -> dict:
        return run_diagnosis(data)

    def suggest_manifolds(self, diagnosis: dict, candidates: list[str] | None = None) -> list[str]:
        return self.recommender.recommend(diagnosis, candidates or self.default_manifolds)

    def test_manifold(self, data, manifold: str, k: int = 8, mask_ratio: float = 0.25, seed: int | None = None) -> dict:
        return evaluate_manifold(data, manifold, k=k, mask_ratio=mask_ratio, seed=seed)

    def explore(
        self,
        data,
        dataset_id: str = "anonymous",
        manifolds: list[str] | None = None,
        k: int = 8,
        mask_ratio: float = 0.25,
        seed: int | None = None,
        save: bool = True,
    ) -> dict:
        diagnosis = self.diagnose(data)
        ordered = self.suggest_manifolds(diagnosis, manifolds)
        evaluations = rank_manifolds(data, ordered, k=k, mask_ratio=mask_ratio, seed=seed)

        if save:
            for evaluation in evaluations:
                signature = experiment_signature(diagnosis, evaluation)
                self.registry.save_experiment(dataset_id, diagnosis, evaluation, signature)

        best = evaluations[0] if evaluations else None
        report = {
            "dataset_id": dataset_id,
            "diagnosis": diagnosis,
            "suggested_manifolds": ordered,
            "evaluations": evaluations,
            "best": best,
        }
        self._last_report = report
        return report

    def report(self) -> dict | None:
        return self._last_report
