"""Main ESO orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eso.data.loader import load_dataset
from eso.db.lab_store import RelationalLabStore
from eso.fingerprints import (
    CapitalProfileSegmenter,
    CascadeAnalyzer,
    compute_microstructure_fingerprints,
)
from eso.diagnostics.report import run_diagnosis
from eso.meta.recommender import HeuristicRecommender
from eso.meta.registry import ExperimentRegistry
from eso.meta.signature import experiment_signature
from eso.meta.soft_selector import FamilyPrioritizedSelector, MethodCrossoverGenerator
from eso.radiography import (
    EntityInfluenceMapper,
    ExogenousCouplingAnalyzer,
    MathematicalRadiographer,
)
from eso.topology.evaluator import evaluate_manifold, rank_manifolds
from eso.topology.manifolds import get_manifold
from eso.topology.validation import validate_manifolds


@dataclass
class ESOExplorer:
    """Diagnose data, test candidate manifolds, and register experiments."""

    registry_path: str = "experiments/eso_registry.csv"
    default_manifolds: list[str] = field(default_factory=lambda: ["circle", "sphere2", "torus2", "cylinder"])
    lab_db_path: str | None = None
    use_soft_selector: bool = False
    epsilon: float = 0.01

    def __post_init__(self):
        self.registry = ExperimentRegistry(self.registry_path)
        self.recommender = HeuristicRecommender()
        self.lab_store = RelationalLabStore(self.lab_db_path) if self.lab_db_path else None
        if self.lab_store is not None:
            self.lab_store.seed_default_arsenal()
        self.soft_selector = FamilyPrioritizedSelector(self.lab_store, epsilon=self.epsilon)
        self.crossover_generator = MethodCrossoverGenerator(self.lab_store)
        self._last_selector_scores: list[dict] = []
        self._last_report = None
        self._last_radiography = None

    def inspect_file(self, path: str, columns: list[str] | None = None, **kwargs) -> dict:
        loaded = load_dataset(path, columns=columns, normalize_method="none", **kwargs)
        return loaded.info()

    def load_file(self, path: str, columns: list[str] | None = None, **kwargs):
        return load_dataset(path, columns=columns, **kwargs)

    def diagnose(self, data) -> dict:
        return run_diagnosis(data)

    def suggest_manifolds(self, diagnosis: dict, candidates: list[str] | None = None) -> list[str]:
        if not self.use_soft_selector:
            return self.recommender.recommend(diagnosis, candidates or self.default_manifolds)

        scores = self.soft_selector.score_candidates(diagnosis, candidates)
        self._last_selector_scores = scores
        ordered: list[str] = []
        for row in scores:
            method_id = str(row["method_id"])
            try:
                get_manifold(method_id)
            except ValueError:
                continue
            ordered.append(method_id)
        return ordered

    def test_manifold(
        self, data, manifold: str, k: int = 8, mask_ratio: float = 0.25,
        seed: int | None = None, projection_method: str = "linear",
    ) -> dict:
        return evaluate_manifold(data, manifold, k=k, mask_ratio=mask_ratio,
                                 seed=seed, projection_method=projection_method)

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
        projection_method: str = "linear",
        proj_neighbors: int = 15,
    ) -> dict:
        diagnosis = self.diagnose(data)
        ordered = self.suggest_manifolds(diagnosis, manifolds)
        if validate:
            evaluations = validate_manifolds(
                data, ordered, k=k, mask_ratio=mask_ratio, n_masks=n_masks, seed=seed,
                projection_method=projection_method, proj_neighbors=proj_neighbors,
            )
        else:
            evaluations = rank_manifolds(
                data, ordered, k=k, mask_ratio=mask_ratio, seed=seed,
                projection_method=projection_method, proj_neighbors=proj_neighbors,
            )

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
            "config": {
                "k": k, "mask_ratio": mask_ratio, "n_masks": n_masks,
                "validate": validate, "projection_method": projection_method,
                "proj_neighbors": proj_neighbors,
            },
        }
        self._last_report = report
        return report

    @staticmethod
    def _structural_summary(raw_df) -> dict | None:
        if not isinstance(raw_df, pd.DataFrame) or not {"close", "volume"}.issubset(raw_df.columns):
            return None

        fingerprints = compute_microstructure_fingerprints(raw_df)
        fingerprint_summary = {
            column: {
                "mean": float(fingerprints[column].mean()),
                "max": float(fingerprints[column].max()),
                "last": float(fingerprints[column].iloc[-1]),
            }
            for column in fingerprints.columns
        }

        cascade = CascadeAnalyzer().analyze(raw_df)
        cascade_summary = {
            key: (
                int(cascade[key])
                if key in {"event_count", "avalanche_count"}
                else float(cascade[key])
            )
            for key in (
                "branching_ratio_eta",
                "event_count",
                "avalanche_count",
                "mean_avalanche_size",
                "mean_avalanche_duration",
                "max_peak_amplification",
            )
        }
        cascade_summary["avalanches"] = list(cascade.get("avalanches", []))

        profile_summary = (
            CapitalProfileSegmenter()
            .summarize(raw_df)
            .reset_index()
            .to_dict(orient="records")
        )
        return {
            "fingerprints": fingerprint_summary,
            "cascade": cascade_summary,
            "capital_profiles": profile_summary,
        }

    @staticmethod
    def _ecosystem_context(data, dataset_id: str, raw_df=None, exogenous_df=None) -> tuple[list[dict], dict | None]:
        if exogenous_df is None:
            return [], None
        if not isinstance(exogenous_df, pd.DataFrame):
            exogenous_df = pd.DataFrame(exogenous_df)

        exog = exogenous_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").reset_index(drop=True)
        if exog.empty:
            return [], None

        if isinstance(raw_df, pd.DataFrame) and "volume" in raw_df.columns:
            target = pd.to_numeric(raw_df["volume"], errors="coerce").reset_index(drop=True)
            target_name = f"{dataset_id}_volume"
        elif isinstance(raw_df, pd.DataFrame) and "close" in raw_df.columns:
            target = pd.to_numeric(raw_df["close"], errors="coerce").pct_change().fillna(0.0).reset_index(drop=True)
            target_name = f"{dataset_id}_returns"
        else:
            array = np.asarray(data, dtype=float)
            target = pd.Series(array[:, 0] if array.ndim > 1 else array, name=dataset_id)
            target_name = str(dataset_id)

        n = min(len(target), len(exog))
        if n < 4:
            return [], None
        target = target.iloc[:n].rename(target_name)
        exog = exog.iloc[:n]

        couplings = ExogenousCouplingAnalyzer().analyze(
            target,
            exog,
            target_name=target_name,
        )

        combined = pd.concat([target.reset_index(drop=True), exog.reset_index(drop=True)], axis=1).dropna()
        graph = None
        if combined.shape[1] >= 2 and len(combined) >= 12:
            try:
                graph = EntityInfluenceMapper().build_graph(combined)
            except ValueError:
                graph = None
        return couplings, graph

    def explore_and_radiograph(
        self,
        data,
        dataset_id: str = "anonymous",
        raw_df=None,
        exogenous_df=None,
        scale_s: str = "unspecified",
        manifolds: list[str] | None = None,
        k: int = 8,
        mask_ratio: float = 0.25,
        n_masks: int = 5,
        seed: int | None = None,
        save: bool = True,
        validate: bool = True,
        projection_method: str = "linear",
        proj_neighbors: int = 15,
        include_crossovers: bool = True,
    ) -> dict:
        report = self.explore(
            data,
            dataset_id=dataset_id,
            manifolds=manifolds,
            k=k,
            mask_ratio=mask_ratio,
            n_masks=n_masks,
            seed=seed,
            save=save,
            validate=validate,
            projection_method=projection_method,
            proj_neighbors=proj_neighbors,
        )

        structural_summary = self._structural_summary(raw_df)
        couplings, influence_graph = self._ecosystem_context(
            data,
            dataset_id,
            raw_df=raw_df,
            exogenous_df=exogenous_df,
        )

        crossovers: list[dict] = []
        if include_crossovers:
            crossover_candidates = report.get("suggested_manifolds") or self.default_manifolds
            crossovers = self.crossover_generator.propose_crossovers(
                crossover_candidates,
                persist=self.lab_store is not None,
            )

        radiography = MathematicalRadiographer().synthesize(
            report,
            scale_s=scale_s,
            ecosystem_couplings=couplings,
            cascade_and_fingerprint_summary=structural_summary,
            dataset_id=dataset_id,
        )
        radiography_payload = radiography.to_dict()

        report["selector_scores"] = list(self._last_selector_scores) if self.use_soft_selector else []
        report["cascade_and_fingerprint_summary"] = structural_summary
        report["ecosystem_couplings"] = couplings
        report["entity_influence_graph"] = influence_graph
        report["crossovers"] = crossovers
        report["radiography"] = radiography_payload
        report["radiography_markdown"] = radiography.to_markdown()

        if self.lab_store is not None:
            array = np.asarray(data)
            space_dim = int(array.shape[1]) if array.ndim > 1 else 1
            config_id = self.lab_store.register_dataset_config(
                dataset_id,
                scale_s,
                space_dim,
                {
                    "k": k,
                    "mask_ratio": mask_ratio,
                    "n_masks": n_masks,
                    "validate": validate,
                    "projection_method": projection_method,
                    "proj_neighbors": proj_neighbors,
                    "use_soft_selector": self.use_soft_selector,
                },
                report.get("diagnosis", {}),
            )
            for evaluation in report.get("evaluations", []):
                method_id = str(evaluation.get("manifold", ""))
                if not method_id or self.lab_store.get_method(method_id) is None:
                    continue
                self.lab_store.record_experiment(
                    dataset_id,
                    method_id,
                    experiment_signature(report.get("diagnosis", {}), evaluation),
                    evaluation,
                    config_id=config_id,
                    relative_to_flat=evaluation.get("relative_to_flat"),
                    oos_stability=evaluation.get("oos_stability"),
                )
            statements = radiography_payload.get("formal_statements_es") or []
            summary_statement = (
                str(statements[0])
                if statements
                else f"Radiografia matematica estructural de {dataset_id} a escala {scale_s}."
            )
            self.lab_store.record_radiography(
                dataset_id,
                summary_statement,
                radiography_payload,
                config_id=config_id,
            )

        self._last_radiography = radiography
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
