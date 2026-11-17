"""Autonomous orchestration for the self-evolving ESO mathematical laboratory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eso.db.lab_store import RelationalLabStore
from eso.diagnostics.report import run_diagnosis
from eso.fingerprints import (
    CapitalProfileSegmenter,
    CascadeAnalyzer,
    compute_microstructure_fingerprints,
)
from eso.meta.signature import experiment_signature
from eso.meta.soft_selector import FamilyPrioritizedSelector, MethodCrossoverGenerator
from eso.radiography import (
    EntityInfluenceMapper,
    ExogenousCouplingAnalyzer,
    MathematicalRadiographer,
)
from eso.topology.manifolds import get_manifold
from eso.topology.validation import validate_manifolds


_REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas values to strict JSON-compatible Python objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(item) for item in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _normalize_numeric_matrix(
    df_or_array: Any,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(df_or_array, pd.DataFrame):
        if feature_columns is not None:
            missing = [column for column in feature_columns if column not in df_or_array.columns]
            if missing:
                raise ValueError(f"feature_columns not present in DataFrame: {missing}")
            numeric = df_or_array.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
        else:
            numeric = df_or_array.select_dtypes(include=[np.number]).copy()
            if numeric.shape[1] == 0:
                numeric = df_or_array.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.dropna(axis=1, how="all")
        labels = [str(column) for column in numeric.columns]
        matrix = numeric.to_numpy(dtype=float)
    else:
        matrix = np.asarray(df_or_array, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError("df_or_array must be a 1D/2D numeric array or pandas DataFrame")
        labels = [f"x{index}" for index in range(matrix.shape[1])]

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("numeric input must contain at least one feature column")
    if matrix.shape[0] < 3:
        raise ValueError("numeric input must contain at least three observations")

    clean = np.asarray(matrix, dtype=float).copy()
    for column in range(clean.shape[1]):
        values = clean[:, column]
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill
        clean[:, column] = values

    mean = clean.mean(axis=0, keepdims=True)
    std = clean.std(axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    normalized = (clean - mean) / std
    return np.nan_to_num(normalized), labels


def _as_ohlcv_frame(value: Any) -> pd.DataFrame | None:
    if not isinstance(value, pd.DataFrame):
        return None
    renamed = value.copy()
    renamed.columns = [str(column).lower() for column in renamed.columns]
    return renamed if _REQUIRED_OHLCV <= set(renamed.columns) else None


def _summary_stats(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "mean": float(numeric.mean()) if len(numeric) else 0.0,
        "max": float(numeric.max()) if len(numeric) else 0.0,
    }


def _oos_stability(evaluation: Mapping[str, Any]) -> float:
    mean = abs(float(_finite_float(evaluation.get("score_mean"), 0.0) or 0.0))
    std = abs(float(_finite_float(evaluation.get("score_std"), 0.0) or 0.0))
    return float(1.0 / (1.0 + std / max(mean, 1e-12)))


def _quality(evaluation: Mapping[str, Any]) -> float:
    score = _finite_float(evaluation.get("score_mean"))
    if score is None:
        return 0.0
    return float(1.0 / (1.0 + max(score, 0.0)))


@dataclass
class AutonomousMathLab:
    """Run sequential ESO laboratory cycles with persistent relational memory."""

    store_path: str | Path = "experiments/eso_lab.sqlite3"
    store: RelationalLabStore | None = None
    epsilon: float = 0.01
    top_k_manifolds: int = 4
    crossover_limit: int = 3
    seed: int = 123

    def __post_init__(self) -> None:
        if not (0.0 < float(self.epsilon) < 1.0):
            raise ValueError("epsilon must satisfy 0 < epsilon < 1")
        if int(self.top_k_manifolds) < 1:
            raise ValueError("top_k_manifolds must be >= 1")
        if int(self.crossover_limit) < 0:
            raise ValueError("crossover_limit must be >= 0")

        self._owns_store = self.store is None
        if self.store is None:
            self.store = RelationalLabStore(self.store_path)
        self.store.seed_default_arsenal()
        self._cycle_index = 0

    def close(self) -> None:
        if self._owns_store and self.store is not None:
            self.store.close()

    def __enter__(self) -> "AutonomousMathLab":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _evaluable(row: Mapping[str, Any]) -> bool:
        try:
            get_manifold(str(row["method_id"]))
            return True
        except (TypeError, ValueError):
            return False

    def _select_manifolds(self, ranking: list[dict[str, Any]]) -> tuple[list[str], set[str]]:
        evaluable = [row for row in ranking if self._evaluable(row)]
        top = evaluable[: int(self.top_k_manifolds)]
        selected = [str(row["method_id"]) for row in top]
        exploration_ids: set[str] = set()

        remaining = [row for row in evaluable if str(row["method_id"]) not in selected]
        if remaining:
            low_count = max(1, len(remaining) // 3)
            low_pool = sorted(
                remaining,
                key=lambda row: (float(row.get("probability", 0.0)), str(row["method_id"])),
            )[:low_count]
            rng = np.random.default_rng(int(self.seed) + self._cycle_index)
            chosen = low_pool[int(rng.integers(0, len(low_pool)))]
            method_id = str(chosen["method_id"])
            selected.append(method_id)
            exploration_ids.add(method_id)

        return selected, exploration_ids

    @staticmethod
    def _target_series(
        df_or_array: Any,
        feature_columns: list[str] | tuple[str, ...] | None,
    ) -> pd.Series:
        if isinstance(df_or_array, pd.DataFrame):
            lower_lookup = {str(column).lower(): column for column in df_or_array.columns}
            if "close" in lower_lookup:
                column = lower_lookup["close"]
            elif feature_columns:
                column = feature_columns[0]
            else:
                numeric = list(df_or_array.select_dtypes(include=[np.number]).columns)
                if not numeric:
                    raise ValueError("no numeric target column is available for exogenous coupling")
                column = numeric[0]
            return pd.to_numeric(df_or_array[column], errors="coerce").rename(str(column))

        matrix = np.asarray(df_or_array, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return pd.Series(matrix[:, 0], name="target")

    def _market_enrichment(
        self,
        df_or_array: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        ohlcv = _as_ohlcv_frame(df_or_array)
        if ohlcv is None:
            return {}, {}, {}

        window = max(3, min(20, len(ohlcv)))
        fingerprints = compute_microstructure_fingerprints(ohlcv, window=window)
        fingerprint_summary = {
            column: _summary_stats(fingerprints[column])
            for column in (
                "stealth_accumulation_index",
                "vpin",
                "kyle_lambda",
                "order_splitting_fingerprint",
            )
        }

        cascade_window = max(3, min(40, len(ohlcv)))
        cascade_raw = CascadeAnalyzer(window=cascade_window).analyze(ohlcv)
        cascade_summary = {
            key: _json_safe(cascade_raw.get(key))
            for key in (
                "branching_ratio_eta",
                "event_count",
                "avalanche_count",
                "avalanches",
                "mean_avalanche_size",
                "mean_avalanche_duration",
                "max_peak_amplification",
            )
        }
        cascade_summary["mean_cascade_risk"] = _summary_stats(
            cascade_raw["rolling_cascade_risk"]
        )["mean"]
        cascade_summary["max_snowball_amplification"] = _summary_stats(
            cascade_raw["multiscale_snowball_amplification"]
        )["max"]

        profile_window = max(3, min(30, len(ohlcv)))
        segmenter = CapitalProfileSegmenter(window=profile_window)
        profiles = segmenter.transform(ohlcv)
        profile_summary = segmenter.summarize(ohlcv)
        capital_profiles = {
            "regime_counts": {
                str(key): int(value)
                for key, value in profiles["capital_regime"].value_counts().to_dict().items()
            },
            "summary": [
                {
                    "capital_regime": str(index),
                    **{str(key): _json_safe(value) for key, value in row.items()},
                }
                for index, row in profile_summary.to_dict(orient="index").items()
            ],
        }
        return fingerprint_summary, cascade_summary, capital_profiles

    def _ecosystem_couplings(
        self,
        df_or_array: Any,
        feature_columns: list[str] | tuple[str, ...] | None,
        peer_series_df: pd.DataFrame | None,
        exogenous_df: pd.DataFrame | pd.Series | Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        couplings: list[dict[str, Any]] = []

        if peer_series_df is not None:
            peer_numeric = peer_series_df.apply(pd.to_numeric, errors="coerce")
            if peer_numeric.shape[1] >= 2 and len(peer_numeric) >= 12:
                mapper = EntityInfluenceMapper().fit(peer_numeric)
                for edge in mapper.to_dict()["edges"]:
                    couplings.append({"source_type": "peer_influence", **dict(edge)})

        if exogenous_df is not None:
            target = self._target_series(df_or_array, feature_columns)
            analyzer = ExogenousCouplingAnalyzer()
            for finding in analyzer.analyze(target, exogenous_df):
                couplings.append({"source_type": "exogenous", **dict(finding)})

        return couplings

    def _operator_evidence(
        self,
        method_id: str,
        selection_probability: float,
        fingerprint_summary: Mapping[str, Any],
        cascade_summary: Mapping[str, Any],
    ) -> float:
        if method_id == "cascade_branching" and cascade_summary:
            eta = float(_finite_float(cascade_summary.get("branching_ratio_eta"), 0.0) or 0.0)
            return float(max(0.0, eta) / (1.0 + max(0.0, eta)))
        if method_id == "microstructure_flow" and fingerprint_summary:
            means = [
                abs(float(_finite_float(item.get("mean"), 0.0) or 0.0))
                for item in fingerprint_summary.values()
                if isinstance(item, Mapping)
            ]
            if means:
                return float(np.mean([math.tanh(value) for value in means]))
        return float(selection_probability)

    def _evaluate_crossovers(
        self,
        ranking: list[dict[str, Any]],
        evaluations: list[dict[str, Any]],
        fingerprint_summary: Mapping[str, Any],
        cascade_summary: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if int(self.crossover_limit) <= 0:
            return []

        evaluated_ids = {str(item["method_id"]) for item in evaluations}
        selected_rows = [row for row in ranking if str(row["method_id"]) in evaluated_ids]
        operator_rows = [row for row in ranking if not self._evaluable(row)]
        candidates = selected_rows + operator_rows
        if len(candidates) < 2:
            return []

        generator = MethodCrossoverGenerator(self.store)
        proposals = generator.propose_crossovers(
            candidates=candidates,
            limit=int(self.crossover_limit),
            persist=True,
        )
        ranking_by_id = {str(row["method_id"]): row for row in ranking}
        evaluation_by_id = {str(row["method_id"]): row for row in evaluations}

        output = []
        for proposal in proposals:
            parent_a = str(proposal["parent_method_a"])
            parent_b = str(proposal["parent_method_b"])
            row_a = ranking_by_id[parent_a]
            row_b = ranking_by_id[parent_b]
            probability_a = float(row_a["probability"])
            probability_b = float(row_b["probability"])

            if parent_a in evaluation_by_id:
                quality_a = _quality(evaluation_by_id[parent_a])
            else:
                quality_a = self._operator_evidence(
                    parent_a, probability_a, fingerprint_summary, cascade_summary
                )
            if parent_b in evaluation_by_id:
                quality_b = _quality(evaluation_by_id[parent_b])
            else:
                quality_b = self._operator_evidence(
                    parent_b, probability_b, fingerprint_summary, cascade_summary
                )

            total_weight = probability_a + probability_b
            hybrid_score = (
                (probability_a * quality_a + probability_b * quality_b) / total_weight
                if total_weight > 0
                else 0.0
            )
            finite = all(math.isfinite(value) for value in (quality_a, quality_b, hybrid_score))
            status = "validated" if finite and hybrid_score > 0.0 else "rejected"
            metrics = {
                "parent_quality_a": quality_a,
                "parent_quality_b": quality_b,
                "parent_probability_a": probability_a,
                "parent_probability_b": probability_b,
                "hybrid_weighted_score": hybrid_score,
                "evaluation_rule": "weighted finite parent evidence; positive hybrid score validates",
            }
            self.store.record_crossover(
                parent_a,
                parent_b,
                str(proposal["emergent_family"]),
                str(proposal["hypothesis"]),
                metrics=metrics,
                status=status,
                crossover_id=str(proposal["crossover_id"]),
            )
            output.append({**proposal, "metrics": metrics, "status": status})
        return output

    def run_cycle(
        self,
        df_or_array: Any,
        dataset_id: str = "anonymous",
        scale_s: str = "unspecified",
        feature_columns: list[str] | tuple[str, ...] | None = None,
        peer_series_df: pd.DataFrame | None = None,
        exogenous_df: pd.DataFrame | pd.Series | Mapping[str, Any] | None = None,
        k: int = 6,
        mask_ratio: float = 0.25,
        n_masks: int = 3,
        projection_method: str = "linear",
    ) -> dict[str, Any]:
        """Run one full autonomous laboratory cycle and persist its evidence."""
        matrix, features = _normalize_numeric_matrix(df_or_array, feature_columns)
        diagnosis = run_diagnosis(matrix)

        selector = FamilyPrioritizedSelector(self.store, epsilon=float(self.epsilon))
        ranking = selector.score_candidates(diagnosis)
        selection_weights = {
            str(row["method_id"]): float(row["probability"])
            for row in ranking
        }
        if not selection_weights or min(selection_weights.values()) + 1e-15 < float(self.epsilon):
            raise RuntimeError("selector violated the positive exploration floor")

        selected_ids, exploration_ids = self._select_manifolds(ranking)
        eval_k = max(1, min(int(k), len(matrix) - 1))
        raw_evaluations = validate_manifolds(
            matrix,
            selected_ids,
            k=eval_k,
            mask_ratio=float(mask_ratio),
            n_masks=int(n_masks),
            seed=int(self.seed) + self._cycle_index * 1000,
            projection_method=str(projection_method),
        )

        canonical_to_id = {
            str(get_manifold(method_id).name): method_id for method_id in selected_ids
        }
        ranking_by_id = {str(row["method_id"]): row for row in ranking}
        evaluations: list[dict[str, Any]] = []
        for evaluation in raw_evaluations:
            method_id = canonical_to_id.get(str(evaluation.get("manifold")))
            if method_id is None:
                continue
            relative = _finite_float(
                evaluation.get("relative_to_flat_mean"),
                _finite_float(evaluation.get("relative_to_flat")),
            )
            stability = _oos_stability(evaluation)
            item = {
                **evaluation,
                "method_id": method_id,
                "selection_probability": float(ranking_by_id[method_id]["probability"]),
                "selection_reason": "exploration" if method_id in exploration_ids else "priority",
                "relative_to_flat": relative,
                "oos_stability": stability,
            }
            evaluations.append(item)

        config_id = self.store.register_dataset_config(
            str(dataset_id),
            str(scale_s),
            int(matrix.shape[1]),
            transform_config={
                "normalization": "zscore_finite_fill",
                "feature_columns": features,
                "projection_method": str(projection_method),
                "k": eval_k,
                "mask_ratio": float(mask_ratio),
                "n_masks": int(n_masks),
            },
            properties={
                "diagnosis": diagnosis,
                "selection_weights": selection_weights,
                "selected_manifolds": selected_ids,
                "exploration_manifolds": sorted(exploration_ids),
            },
        )

        for evaluation in evaluations:
            method_id = str(evaluation["method_id"])
            persisted_metrics = {
                key: value
                for key, value in evaluation.items()
                if key not in {"method_id", "selection_reason", "selection_probability", "score"}
            }
            persisted_metrics["quality"] = _quality(evaluation)
            self.store.record_experiment(
                str(dataset_id),
                method_id,
                experiment_signature(diagnosis, evaluation),
                _json_safe(persisted_metrics),
                config_id=config_id,
                relative_to_flat=_finite_float(evaluation.get("relative_to_flat")),
                oos_stability=_finite_float(evaluation.get("oos_stability")),
            )

        fingerprint_summary, cascade_summary, capital_profiles = self._market_enrichment(df_or_array)
        ecosystem_couplings = self._ecosystem_couplings(
            df_or_array,
            feature_columns,
            peer_series_df,
            exogenous_df,
        )
        crossovers = self._evaluate_crossovers(
            ranking,
            evaluations,
            fingerprint_summary,
            cascade_summary,
        )

        best = min(
            evaluations,
            key=lambda item: (
                float(_finite_float(item.get("score_mean"), float("inf")) or float("inf")),
                str(item["method_id"]),
            ),
        ) if evaluations else {}
        cascade_and_fingerprint_summary = {
            "fingerprints": fingerprint_summary,
            "cascade": cascade_summary,
            "capital_profiles": capital_profiles,
        }
        radiografia = MathematicalRadiographer(language="both").synthesize(
            {
                "dataset_id": str(dataset_id),
                "diagnosis": diagnosis,
                "best": best,
                "evaluations": evaluations,
                "ambient_space_dim_j": int(matrix.shape[1]),
            },
            scale_s=str(scale_s),
            ecosystem_couplings=ecosystem_couplings,
            cascade_and_fingerprint_summary=cascade_and_fingerprint_summary,
            dataset_id=str(dataset_id),
        )
        radiography_payload = radiografia.to_dict()
        statements = radiografia.formal_statements_es or radiografia.formal_statements_en
        summary_statement = (
            statements[0]
            if statements
            else f"Mathematical radiography for {dataset_id} at scale {scale_s}."
        )
        radiography_id = self.store.record_radiography(
            str(dataset_id),
            str(summary_statement),
            radiography_payload,
            config_id=config_id,
        )

        result = {
            "dataset_id": str(dataset_id),
            "config_id": config_id,
            "radiography_id": radiography_id,
            "selection_weights": selection_weights,
            "selection_ranking": ranking,
            "evaluations": evaluations,
            "crossovers": crossovers,
            "cascade_summary": cascade_summary,
            "fingerprint_summary": fingerprint_summary,
            "capital_profiles": capital_profiles,
            "ecosystem_couplings": ecosystem_couplings,
            "radiography": radiography_payload,
        }
        self._cycle_index += 1
        return _json_safe(result)

    def evolve_multiple(
        self,
        datasets: list[dict[str, Any]],
        **cycle_kwargs: Any,
    ) -> dict[str, Any]:
        """Run datasets sequentially so earlier experiments reinforce later selection."""
        cycles: list[dict[str, Any]] = []
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, Mapping):
                raise TypeError("each dataset entry must be a mapping")
            item = dict(dataset)
            data = item.pop("df_or_array", item.pop("data", None))
            if data is None:
                raise ValueError("each dataset entry requires 'df_or_array' or 'data'")
            item.setdefault("dataset_id", f"dataset-{index + 1}")
            args = {**cycle_kwargs, **item}
            cycles.append(self.run_cycle(data, **args))

        return _json_safe(
            {
                "cycle_count": len(cycles),
                "cycles": cycles,
                "family_performance": self.store.query_family_performance(),
            }
        )
