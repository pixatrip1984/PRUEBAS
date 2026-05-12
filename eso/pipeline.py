"""Main ESO orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from eso.data.loader import load_dataset
from eso.diagnostics.report import run_diagnosis
from eso.meta.recommender import HeuristicRecommender
from eso.meta.registry import ExperimentRegistry
from eso.meta.signature import experiment_signature
from eso.topology.evaluator import evaluate_manifold, rank_manifolds
from eso.topology.validation import validate_manifolds


@dataclass
class ESOExplorer:
    """Diagnose data, test candidate manifolds, and register experiments."""

    registry_path: str = "experiments/eso_registry.csv"
    default_manifolds: list[str] = field(default_factory=lambda: ["circle", "sphere2", "torus2", "cylinder"])

    def __post_init__(self):
        self.registry = ExperimentRegistry(self.registry_path)
        self.recommender = HeuristicRecommender()
        self._last_report = None

    def inspect_file(self, path: str, columns: list[str] | None = None, **kwargs) -> dict:
        loaded = load_dataset(path, columns=columns, normalize_method="none", **kwargs)
        return loaded.info()

    def load_file(self, path: str, columns: list[str] | None = None, **kwargs):
        return load_dataset(path, columns=columns, **kwargs)

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
        n_masks: int = 5,
        seed: int | None = None,
        save: bool = True,
        validate: bool = True,
    ) -> dict:
        diagnosis = self.diagnose(data)
        ordered = self.suggest_manifolds(diagnosis, manifolds)
        if validate:
            evaluations = validate_manifolds(data, ordered, k=k, mask_ratio=mask_ratio, n_masks=n_masks, seed=seed)
        else:
            evaluations = rank_manifolds(data, ordered, k=k, mask_ratio=mask_ratio, seed=seed)

        for rank, evaluation in enumerate(evaluations, start=1):
            evaluation["rank"] = rank

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
            "config": {"k": k, "mask_ratio": mask_ratio, "n_masks": n_masks, "validate": validate},
        }
        self._last_report = report
        return report

    def explore_file(
        self,
        path: str,
        dataset_id: str | None = None,
        columns: list[str] | None = None,
        output_dir: str | None = None,
        manifolds: list[str] | None = None,
        normalize_method: str = "robust",
        max_rows: int | None = None,
        window_size: int | None = None,
        window_step: int = 1,
        window_mode: str = "last",
        k: int = 8,
        mask_ratio: float = 0.25,
        n_masks: int = 5,
        seed: int | None = 123,
        save: bool = True,
    ) -> dict:
        loaded = self.load_file(
            path,
            columns=columns,
            normalize_method=normalize_method,
            max_rows=max_rows,
            window_size=window_size,
            window_step=window_step,
            window_mode=window_mode,
        )
        dataset_id = dataset_id or Path(path).stem
        report = self.explore(
            loaded.data,
            dataset_id=dataset_id,
            manifolds=manifolds,
            k=k,
            mask_ratio=mask_ratio,
            n_masks=n_masks,
            seed=seed,
            save=save,
            validate=True,
        )
        report["dataset"] = loaded.info()
        return report

    def report(self) -> dict | None:
        return self._last_report
