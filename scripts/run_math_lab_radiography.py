"""Empirical BTC and multi-asset Mathematical Radiography runner.

This module intentionally depends only on Wave-1 ESO components so it can run
independently of sibling Wave-2 branches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eso.data.features import build_financial_features, feature_column_groups
from eso.data.loader import load_dataset
from eso.db.lab_store import RelationalLabStore
from eso.diagnostics.report import run_diagnosis
from eso.fingerprints import (
    CapitalProfileSegmenter,
    CascadeAnalyzer,
    compute_microstructure_fingerprints,
)
from eso.meta.soft_selector import FamilyPrioritizedSelector, MethodCrossoverGenerator
from eso.radiography import (
    EntityInfluenceMapper,
    ExogenousCouplingAnalyzer,
    MathematicalRadiographer,
)
from eso.topology.manifolds import get_manifold
from eso.topology.validation import validate_manifolds


_STANDARD_MARKET_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "n_trades",
    "vwap",
    "volume_usd",
    "taker_buy_volume",
    "taker_sell_volume",
    "delta",
}


def _jsonable(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(item) for item in value.reset_index().to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _strict_json(payload: Any) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)


def _asset_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_4h_full", "_4h"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by UTC timestamps when a timestamp column exists."""
    out = df.copy()
    if "timestamp" not in out.columns:
        out.index = pd.RangeIndex(len(out))
        return out

    raw = out["timestamp"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        median = float(numeric.abs().median()) if numeric.notna().any() else 0.0
        if median >= 1e17:
            unit = "ns"
        elif median >= 1e14:
            unit = "us"
        elif median >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        index = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        index = pd.to_datetime(raw, utc=True, errors="coerce")

    out = out.loc[index.notna()].copy()
    out.index = pd.DatetimeIndex(index[index.notna()], name="timestamp")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _read_asset(path: Path, max_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if max_rows is not None and max_rows > 0 and len(frame) > max_rows:
        frame = frame.tail(int(max_rows)).copy()
    return _timestamp_index(frame)


def _discover_assets(data_dir: Path) -> tuple[Path, list[Path]]:
    btc_files = sorted(data_dir.glob("BTCUSDT*.csv"))
    if not btc_files:
        raise FileNotFoundError(f"No BTCUSDT*.csv dataset found in {data_dir}")
    exact = data_dir / "BTCUSDT_4h_full.csv"
    primary = exact if exact in btc_files else btc_files[0]
    altcoins = [
        path
        for path in sorted(data_dir.glob("*_4h.csv"))
        if path != primary and not path.name.startswith("BTCUSDT")
    ]
    return primary, altcoins


def _fingerprint_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        summary[str(column)] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "last": float(values.iloc[-1]) if len(values) else 0.0,
        }
    return summary


def _cascade_summary(result: dict[str, Any]) -> dict[str, Any]:
    rolling = pd.to_numeric(result.get("rolling_cascade_risk", pd.Series(dtype=float)), errors="coerce")
    snowball = pd.to_numeric(
        result.get("multiscale_snowball_amplification", pd.Series(dtype=float)),
        errors="coerce",
    )
    return {
        "branching_ratio_eta": float(result.get("branching_ratio_eta", 0.0)),
        "event_count": int(result.get("event_count", 0)),
        "avalanche_count": int(result.get("avalanche_count", 0)),
        "avalanches": _jsonable(result.get("avalanches", [])),
        "mean_avalanche_size": float(result.get("mean_avalanche_size", 0.0)),
        "mean_avalanche_duration": float(result.get("mean_avalanche_duration", 0.0)),
        "max_peak_amplification": float(result.get("max_peak_amplification", 0.0)),
        "rolling_cascade_risk": {
            "mean": float(rolling.mean()) if len(rolling) else 0.0,
            "max": float(rolling.max()) if len(rolling) else 0.0,
            "last": float(rolling.iloc[-1]) if len(rolling) else 0.0,
        },
        "multiscale_snowball_amplification": {
            "mean": float(snowball.mean()) if len(snowball) else 0.0,
            "max": float(snowball.max()) if len(snowball) else 0.0,
            "last": float(snowball.iloc[-1]) if len(snowball) else 0.0,
        },
    }


def _capital_summary(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    named = summary.copy()
    named.index.name = "capital_regime"
    return _jsonable(named.reset_index().to_dict(orient="records"))


def _log_return(frame: pd.DataFrame) -> pd.Series | None:
    if "close" not in frame.columns:
        return None
    close = pd.to_numeric(frame["close"], errors="coerce").replace(0.0, np.nan)
    return np.log(close).diff()


def _volume_flow(frame: pd.DataFrame) -> pd.Series | None:
    if "volume" not in frame.columns:
        return None
    volume = pd.to_numeric(frame["volume"], errors="coerce").clip(lower=0.0)
    return np.log1p(volume).diff()


def _ecosystem_signals(
    primary_name: str,
    primary: pd.DataFrame,
    alts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    series: list[pd.Series] = []
    for name, frame in [(primary_name, primary), *alts.items()]:
        returns = _log_return(frame)
        if returns is not None:
            series.append(returns.rename(f"{name}.log_return"))
        flow = _volume_flow(frame)
        if flow is not None:
            series.append(flow.rename(f"{name}.volume_flow"))
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join="outer").sort_index()


def _exogenous_frame(alts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series: list[pd.Series] = []
    for name, frame in alts.items():
        for column in frame.columns:
            label = str(column)
            if label in _STANDARD_MARKET_COLUMNS:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                series.append(values.rename(f"{name}.{label}"))
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join="outer").sort_index()


def _primary_targets(primary_name: str, primary: pd.DataFrame) -> dict[str, pd.Series]:
    targets: dict[str, pd.Series] = {}
    returns = _log_return(primary)
    if returns is not None:
        targets[f"{primary_name}.log_return"] = returns
    if "volume" in primary.columns:
        targets[f"{primary_name}.volume"] = pd.to_numeric(primary["volume"], errors="coerce")
    if "delta" in primary.columns:
        targets[f"{primary_name}.delta"] = pd.to_numeric(primary["delta"], errors="coerce")
    return targets


def _valid_manifold_ids(ranked: Iterable[dict[str, Any]]) -> list[str]:
    valid: list[str] = []
    for item in ranked:
        method_id = str(item["method_id"])
        try:
            get_manifold(method_id)
        except (TypeError, ValueError):
            continue
        valid.append(method_id)
    return valid


def _experiment_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "score",
        "score_mean",
        "score_std",
        "reconstruction_error",
        "reconstruction_error_mean",
        "reconstruction_error_std",
        "reconstruction_error_ci95",
        "relative_to_flat",
        "relative_to_flat_mean",
        "relative_to_flat_std",
        "smoothness",
        "latent_utilization",
        "complexity_penalty",
        "flat_baseline_error",
        "n_points",
        "ambient_dim",
        "manifold_dim",
        "n_masks",
    )
    return {key: _jsonable(evaluation[key]) for key in keys if key in evaluation}


def _persist_crossovers(
    store: RelationalLabStore,
    generator: MethodCrossoverGenerator,
    selector_rows: list[dict[str, Any]],
    experiment_by_method: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    method_ids = [str(item["method_id"]) for item in selector_rows]
    proposals = generator.propose_crossovers(method_ids, limit=max(1, int(limit)), persist=False)
    weights = {str(item["method_id"]): float(item["probability"]) for item in selector_rows}
    persisted: list[dict[str, Any]] = []
    for proposal in proposals:
        a = str(proposal["parent_method_a"])
        b = str(proposal["parent_method_b"])
        parent_results = [experiment_by_method[m] for m in (a, b) if m in experiment_by_method]
        relative = [
            float(item.get("relative_to_flat_mean", item.get("relative_to_flat")))
            for item in parent_results
            if item.get("relative_to_flat_mean", item.get("relative_to_flat")) is not None
            and math.isfinite(float(item.get("relative_to_flat_mean", item.get("relative_to_flat"))))
        ]
        metrics = {
            "evaluation_kind": "parent_evidence_proxy",
            "validated_parent_count": len(parent_results),
            "selector_probability_mean": float(np.mean([weights.get(a, 0.0), weights.get(b, 0.0)])),
            "parent_relative_to_flat_mean": float(np.mean(relative)) if relative else None,
            "falsifiable_next_step": (
                "Instantiate a valid crossover operator and require improvement over the stronger "
                "parent without a validation-stability regression."
            ),
        }
        store.record_crossover(
            a,
            b,
            str(proposal["emergent_family"]),
            str(proposal["hypothesis"]),
            metrics=metrics,
            status="evaluated",
            crossover_id=str(proposal["crossover_id"]),
        )
        item = dict(proposal)
        item["metrics"] = metrics
        item["status"] = "evaluated"
        persisted.append(item)
    return persisted


def _markdown_report(
    report: dict[str, Any],
    radiography_markdown: str,
) -> str:
    lines = [
        radiography_markdown,
        "",
        "## Empirical discovery / Descubrimiento empírico",
        "",
        f"- Primary dataset: '{report['dataset']['primary_file']}'",
        f"- Rows analyzed: {report['dataset']['primary_rows']}",
        f"- Feature mode: '{report['dataset']['feature_mode']}'",
        f"- Normalized features: {', '.join(report['dataset']['feature_columns'])}",
        "",
        "### Non-discarding selector",
        "",
        "| rank | method | family | probability |",
        "|---:|---|---|---:|",
    ]
    for item in report["selector_probabilities"]:
        lines.append(
            f"| {item['rank']} | {item['method_id']} | {item['family']} | {float(item['probability']):.6f} |"
        )

    lines += [
        "",
        "### Evaluated topology manifolds",
        "",
        "| manifold | score_mean | relative_to_flat_mean |",
        "|---|---:|---:|",
    ]
    for item in report["evaluated_manifolds"]:
        lines.append(
            f"| {item.get('manifold', 'unknown')} | "
            f"{float(item.get('score_mean') or 0.0):.6g} | "
            f"{float(item.get('relative_to_flat_mean', item.get('relative_to_flat')) or 0.0):.6g} |"
        )

    lines += ["", "### Microstructure, cascades and capital profiles", ""]
    lines.append(
        f"- Cascade branching ratio eta: {float(report['cascade']['branching_ratio_eta']):.6g}; "
        f"avalanches: {report['cascade']['avalanche_count']}."
    )
    for name, stats in report["microstructure_fingerprints"].items():
        lines.append(
            f"- '{name}' mean={float(stats['mean']):.6g}, max={float(stats['max']):.6g}, "
            f"last={float(stats['last']):.6g}."
        )
    for item in report["capital_profiles"]:
        lines.append(
            f"- Capital regime '{item['capital_regime']}': observations={item['observations']}, "
            f"mean_confidence={float(item['mean_confidence']):.4f}."
        )

    lines += ["", "### Ecosystem influence graph", ""]
    edges = report["ecosystem_graph"].get("edges", [])
    if edges:
        for edge in edges:
            lines.append(
                f"- {edge['source']} → {edge['target']}: sign={edge['sign']}, "
                f"lag={edge['best_lag']}, |r|={float(edge['strength']):.4f}."
            )
    else:
        lines.append("- No edge crossed the empirical influence threshold in the bounded sample.")

    lines += ["", "### Exogenous coupling regimes (Z)", ""]
    if report["exogenous_regimes"]:
        for item in report["exogenous_regimes"]:
            periods = ", ".join(str(x) for x in item.get("periods", []))
            lines.append(
                f"- {item['indicator']} ↔ {item['target']}: regime={item['regime']}, "
                f"lag={item['best_lag']}, r={float(item['peak_corr']):.4f}, Z={periods}."
            )
    else:
        lines.append("- No exogenous regime crossed the threshold for the available columns.")

    lines += ["", "### Cross-family crossover hypotheses", ""]
    for item in report["crossovers"]:
        lines.append(
            f"- '{item['parent_method_a']} × {item['parent_method_b']}' → "
            f"'{item['emergent_family']}' ({item['status']})."
        )
    return "\n".join(lines) + "\n"


def run_empirical_math_lab(
    data_dir: str | Path = "data",
    output_dir: str | Path = "reports/math_lab",
    db_path: str | Path = "experiments/eso_lab.sqlite3",
    max_rows: int = 1000,
    scale_s: str = "4h",
    feature_mode: str = "microstructure",
    k: int = 6,
    n_masks: int = 3,
    seed: int = 123,
    epsilon: float = 0.01,
) -> dict:
    """Run bounded empirical Mathematical Radiography over BTC and available altcoins."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    database_path = Path(db_path)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if n_masks <= 0:
        raise ValueError("n_masks must be positive")

    primary_path, alt_paths = _discover_assets(data_path)
    primary_name = _asset_name(primary_path)
    primary = _read_asset(primary_path, int(max_rows))
    if "close" not in primary.columns:
        raise ValueError(f"Primary dataset {primary_path} must contain a close column")

    alts = {_asset_name(path): _read_asset(path, int(max_rows)) for path in alt_paths}

    features = build_financial_features(primary)
    groups = feature_column_groups()
    requested = groups.get(feature_mode, groups.get("full", []))
    feature_columns = [column for column in requested if column in features.columns]
    if not feature_columns:
        feature_columns = [
            column
            for column in features.columns
            if pd.api.types.is_numeric_dtype(features[column])
        ]
    if not feature_columns:
        raise ValueError("No usable financial features were produced for structural diagnosis")

    loaded = load_dataset(
        features,
        columns=feature_columns,
        normalize_method="robust",
    )
    normalized = np.asarray(loaded.data, dtype=float)
    if normalized.ndim != 2 or len(normalized) < 12:
        raise ValueError("At least 12 normalized feature rows are required for empirical radiography")

    dataset_id = f"{primary_name}-{scale_s}"
    diagnosis = run_diagnosis(normalized)

    fingerprint_window = max(3, min(20, max(3, len(primary) // 4)))
    fingerprints = compute_microstructure_fingerprints(primary, window=fingerprint_window)
    fingerprint_stats = _fingerprint_summary(fingerprints)
    cascade = _cascade_summary(CascadeAnalyzer().analyze(primary))
    capital = _capital_summary(CapitalProfileSegmenter().summarize(primary))

    signals = _ecosystem_signals(primary_name, primary, alts)
    if signals.shape[1] >= 2:
        ecosystem_window = max(24, min(80, max(24, len(signals) // 3)))
        mapper = EntityInfluenceMapper(
            max_lag=6,
            threshold=0.55,
            dominance_ratio=1.20,
            window_size=ecosystem_window,
        ).fit(signals)
        ecosystem_graph = mapper.to_dict()
    else:
        ecosystem_graph = {
            "nodes": list(signals.columns),
            "edges": [],
            "relationships": [],
            "adjacency": {str(column): [] for column in signals.columns},
            "lag_convention": "corr(source[t], target[t + lag]); positive lag means source leads target",
        }

    exogenous = _exogenous_frame(alts)
    exogenous_findings: list[dict[str, Any]] = []
    if not exogenous.empty:
        for target_name, target in _primary_targets(primary_name, primary).items():
            aligned_count = len(pd.concat([exogenous, target.rename(target_name)], axis=1).dropna(how="all"))
            window = max(24, min(72, max(24, aligned_count // 3)))
            analyzer = ExogenousCouplingAnalyzer(
                max_lag=6,
                threshold=0.55,
                window_size=window,
                step=max(4, window // 5),
            )
            exogenous_findings.extend(analyzer.analyze(target.rename(target_name), exogenous))

    output_path.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with RelationalLabStore(database_path) as store:
        store.seed_default_arsenal()
        methods = store.list_methods()
        if epsilon * len(methods) > 1.0 + 1e-12:
            raise ValueError(
                f"epsilon={epsilon} cannot preserve a positive floor across {len(methods)} methods"
            )
        config_id = store.register_dataset_config(
            dataset_id,
            scale_s,
            int(normalized.shape[1]),
            {
                "feature_mode": feature_mode,
                "feature_columns": list(loaded.columns),
                "normalization": loaded.normalized,
                "max_rows": int(max_rows),
            },
            {
                "primary_file": primary_path.name,
                "altcoin_files": [path.name for path in alt_paths],
                "primary_rows": int(len(primary)),
            },
        )

        selector = FamilyPrioritizedSelector(store, epsilon=epsilon)
        all_method_ids = [str(item["method_id"]) for item in methods]
        ranked = selector.prioritize(diagnosis, all_method_ids)
        probabilities = [float(item["probability"]) for item in ranked]
        if not probabilities or min(probabilities) + 1e-15 < float(epsilon):
            raise RuntimeError("FamilyPrioritizedSelector violated the non-discarding epsilon floor")

        evaluable = _valid_manifold_ids(ranked)
        selected = evaluable[: min(int(k), len(evaluable))]
        if not selected:
            raise RuntimeError("No seeded method maps to a valid topology manifold")

        validation_k = min(8, max(2, len(normalized) - 1))
        evaluated = validate_manifolds(
            normalized,
            selected,
            k=validation_k,
            n_masks=int(n_masks),
            seed=int(seed),
            projection_method="linear",
        )

        canonical_to_method = {get_manifold(method_id).name: method_id for method_id in selected}
        experiment_by_method: dict[str, dict[str, Any]] = {}
        for evaluation in evaluated:
            method_id = canonical_to_method.get(str(evaluation.get("manifold")))
            if method_id is None:
                continue
            experiment_by_method[method_id] = evaluation
            relative = evaluation.get("relative_to_flat_mean", evaluation.get("relative_to_flat"))
            score_std = evaluation.get("score_std")
            stability = None
            if score_std is not None and math.isfinite(float(score_std)):
                stability = 1.0 / (1.0 + max(0.0, float(score_std)))
            store.record_experiment(
                dataset_id,
                method_id,
                {
                    "diagnosis_dimension": _jsonable(diagnosis.get("dimension", {})),
                    "feature_mode": feature_mode,
                    "feature_columns": list(loaded.columns),
                },
                _experiment_metrics(evaluation),
                config_id=config_id,
                relative_to_flat=float(relative) if relative is not None and math.isfinite(float(relative)) else None,
                oos_stability=stability,
            )

        generator = MethodCrossoverGenerator(store)
        crossovers = _persist_crossovers(
            store,
            generator,
            ranked,
            experiment_by_method,
            limit=max(3, min(12, int(k) * 2)),
        )

        best = evaluated[0] if evaluated else {}
        exploration_report = {
            "dataset_id": dataset_id,
            "diagnosis": diagnosis,
            "evaluations": evaluated,
            "best": best,
        }
        cascade_and_fingerprints = {
            "cascade": cascade,
            "fingerprints": fingerprint_stats,
            "capital_profiles": capital,
        }
        radiography = MathematicalRadiographer(language="both").synthesize(
            exploration_report,
            scale_s=scale_s,
            ecosystem_couplings=[
                *ecosystem_graph.get("edges", []),
                *exogenous_findings,
            ],
            cascade_and_fingerprint_summary=cascade_and_fingerprints,
            dataset_id=dataset_id,
        )
        radiography_payload = radiography.to_dict()
        summary_statement = (
            radiography.formal_statements_es[0]
            if radiography.formal_statements_es
            else (
                radiography.formal_statements_en[0]
                if radiography.formal_statements_en
                else f"Empirical Mathematical Radiography for {dataset_id}"
            )
        )
        radiography_id = store.record_radiography(
            dataset_id,
            summary_statement,
            radiography_payload,
            config_id=config_id,
        )

    report = {
        "dataset": {
            "dataset_id": dataset_id,
            "primary_file": primary_path.name,
            "altcoin_files": [path.name for path in alt_paths],
            "primary_rows": int(len(primary)),
            "scale_s": scale_s,
            "feature_mode": feature_mode,
            "feature_columns": list(loaded.columns),
        },
        "dataset_config_id": config_id,
        "radiography_id": radiography_id,
        "diagnosis": diagnosis,
        "selector_probabilities": ranked,
        "selector_weights": {
            str(item["method_id"]): float(item["probability"]) for item in ranked
        },
        "min_selector_probability": min(float(item["probability"]) for item in ranked),
        "epsilon_floor": float(epsilon),
        "evaluated_manifolds": evaluated,
        "crossovers": crossovers,
        "microstructure_fingerprints": fingerprint_stats,
        "cascade": cascade,
        "cascade_stats": cascade,
        "capital_profiles": capital,
        "ecosystem_graph": ecosystem_graph,
        "influence_edges": ecosystem_graph.get("edges", []),
        "exogenous_regimes": exogenous_findings,
        "exogenous_findings": exogenous_findings,
        "radiography": radiography_payload,
        "formal_statements": {
            "es": list(radiography.formal_statements_es),
            "en": list(radiography.formal_statements_en),
        },
    }

    json_path = output_path / "empirical_radiography.json"
    md_path = output_path / "empirical_radiography.md"
    json_path.write_text(_strict_json(report) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(_jsonable(report), radiography.to_markdown()), encoding="utf-8")
    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "sqlite": str(database_path),
    }
    return _jsonable(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Empirical multi-asset BTC Mathematical Radiography runner"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/math_lab")
    parser.add_argument("--db-path", default="experiments/eso_lab.sqlite3")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--scale", default="4h")
    parser.add_argument("--feature-mode", default="microstructure")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    result = run_empirical_math_lab(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        db_path=args.db_path,
        max_rows=args.max_rows,
        scale_s=args.scale,
        feature_mode=args.feature_mode,
        seed=args.seed,
    )
    print(result["artifacts"]["json"])
    print(result["artifacts"]["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
