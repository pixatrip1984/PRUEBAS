"""Command line interface for ESO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eso.benchmark import run_benchmark_suite, write_benchmark_result
from eso.contracts import validate_contract
from eso.data.features import build_financial_features, feature_column_groups, prepare_btc_klines
from eso.data.loader import load_dataset, read_table
from eso.data.validation import validate_dataframe
from eso.diagnostics.report import run_diagnosis
from eso.lab import (
    dissect_feature_frame,
    run_universe_dissect,
    write_asset_dissect_bundle,
    write_universe_dissect_result,
)
from eso.lab.context import append_reference_context
from eso.modeling import (
    build_feature_frame,
    compare_model_runs,
    evaluate_supervised_task,
    list_model_runs,
    save_model_run,
    write_model_run,
)
from eso.pipeline import ESOExplorer
from eso.reporting import write_report_bundle


_FEATURE_MODES = [
    "raw", "returns_only", "returns_vol", "microstructure",
    "derivatives", "compact", "alt_funding", "full",
]
_MODEL_FEATURE_MODES = _FEATURE_MODES + ["cycle"]


def _columns(values):
    return values if values else None


def _apply_feature_mode(df, feature_mode: str | None, columns: list[str] | None):
    """Optionally transform df using financial features before ESO processing."""
    if not feature_mode or feature_mode == "raw":
        return df, columns
    feat = build_financial_features(df)
    groups = feature_column_groups()
    if feature_mode in groups:
        cols = [c for c in groups[feature_mode] if c in feat.columns]
    elif feature_mode == "full":
        cols = [c for c in feat.columns]
    else:
        raise ValueError(f"Unknown feature-mode '{feature_mode}'. Choose: raw, " + ", ".join(groups))
    return feat, cols


def cmd_inspect(args) -> int:
    df = read_table(args.path)
    if args.max_rows:
        df = df.head(args.max_rows)
    report = validate_dataframe(df, columns=_columns(args.columns)).to_dict()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _maybe_klines(df, klines_format: bool) -> "pd.DataFrame":
    import pandas as pd
    if klines_format:
        return prepare_btc_klines(df)
    return df


def _slice_rows(df, skip_rows, max_rows):
    if skip_rows:
        df = df.iloc[int(skip_rows):]
    if max_rows:
        df = df.head(int(max_rows))
    return df.reset_index(drop=True)


def cmd_diagnose(args) -> int:
    raw = read_table(args.path)
    raw = _maybe_klines(raw, getattr(args, "klines_format", False))
    raw = _slice_rows(raw, getattr(args, "skip_rows", None), args.max_rows)
    raw, cols = _apply_feature_mode(raw, getattr(args, "feature_mode", None), _columns(args.columns))
    loaded = load_dataset(
        raw,
        columns=cols,
        normalize_method=args.normalize,
        window_size=args.window_size,
        window_step=args.window_step,
        window_mode=args.window_mode,
    )
    report = run_diagnosis(loaded.data)
    out = {"dataset": loaded.info(), "diagnosis": report, "feature_mode": getattr(args, "feature_mode", "raw")}
    print(report.get("summary", "diagnosis complete"))
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def cmd_explore(args) -> int:
    explorer = ESOExplorer(registry_path=args.registry)
    raw = read_table(args.path)
    raw = _maybe_klines(raw, getattr(args, "klines_format", False))
    raw = _slice_rows(raw, getattr(args, "skip_rows", None), args.max_rows)
    raw, cols = _apply_feature_mode(raw, getattr(args, "feature_mode", None), _columns(args.columns))
    loaded = load_dataset(
        raw,
        columns=cols,
        normalize_method=args.normalize,
        window_size=args.window_size,
        window_step=args.window_step,
        window_mode=args.window_mode,
    )
    feature_mode = getattr(args, "feature_mode", "raw") or "raw"
    proj_method = getattr(args, "projection_method", "linear") or "linear"
    proj_nbrs = getattr(args, "proj_neighbors", 15) or 15
    stem = Path(args.path).stem
    proj_tag = f"_{proj_method}" if proj_method != "linear" else ""
    dataset_id = args.dataset_id or (
        f"{stem}_{feature_mode}{proj_tag}" if (feature_mode != "raw" or proj_method != "linear")
        else stem
    )
    report = explorer.explore(
        loaded.data,
        dataset_id=dataset_id,
        manifolds=args.manifolds,
        k=args.k,
        mask_ratio=args.mask_ratio,
        n_masks=args.n_masks,
        seed=args.seed,
        save=not args.no_registry,
        validate=True,
        projection_method=proj_method,
        proj_neighbors=proj_nbrs,
    )
    report["dataset"] = loaded.info()
    report["feature_mode"] = feature_mode
    report["projection_method"] = proj_method
    artifacts = write_report_bundle(report, loaded.data, args.output)
    print(json.dumps({"best": report.get("best"), "artifacts": artifacts}, indent=2, sort_keys=True))
    return 0


def cmd_model_eval(args) -> int:
    raw = read_table(args.path)
    raw = _maybe_klines(raw, getattr(args, "klines_format", False))
    raw = _slice_rows(raw, getattr(args, "skip_rows", None), args.max_rows)
    feature_frame = build_feature_frame(
        raw,
        feature_mode=args.feature_mode,
        columns=_columns(args.columns),
        close_col=args.close_col,
        umap_train_size=args.umap_train_size,
        smooth_windows=tuple(args.smooth_windows),
    )
    dataset_id = args.dataset_id or Path(args.path).stem
    manifest = evaluate_supervised_task(
        feature_frame,
        task=args.task,
        horizon=args.horizon,
        feature_cols=_columns(args.feature_columns),
        train_size=args.train_size,
        train_fraction=args.train_fraction,
        model_type=args.model_type,
        dataset_id=dataset_id,
    )
    output = Path(args.output)
    output_path = output if output.suffix.lower() == ".json" else output / "model_run_manifest.json"
    write_model_run(manifest, output_path)
    registry_row = None
    if not args.no_registry:
        registry_row = save_model_run(manifest, args.registry, output_path)
    print(json.dumps({
        "model_run": str(output_path),
        "registry": None if args.no_registry else args.registry,
        "run_id": None if registry_row is None else registry_row.get("run_id"),
        "task": manifest["task"],
        "target": manifest["target"],
        "split": manifest["split"],
        "primary_metric": manifest["primary_metric"],
        "metrics": manifest["metrics"],
    }, indent=2, sort_keys=True))
    return 0


def cmd_model_runs(args) -> int:
    if args.action == "list":
        rows = list_model_runs(
            registry_path=args.registry,
            limit=args.limit,
            task=args.task,
            dataset_id=args.dataset_id,
        )
        print(json.dumps({"registry": args.registry, "runs": rows}, indent=2, sort_keys=True))
        return 0
    if args.action == "compare":
        if len(args.refs) != 2:
            raise ValueError("model-runs compare requires exactly two run ids or manifest paths")
        result = compare_model_runs(args.refs[0], args.refs[1], registry_path=args.registry)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unknown model-runs action: {args.action}")


def cmd_validate_json(args) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate_contract(payload, schema_version=args.schema_version)
    result = {"path": args.path, "valid": not errors, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def cmd_benchmark(args) -> int:
    result = run_benchmark_suite(
        args.suite,
        output_dir=args.output,
        registry_path=args.registry,
        save_registry=not args.no_registry,
    )
    result_path = Path(args.output) / "benchmark_result.json"
    write_benchmark_result(result, result_path)
    print(json.dumps({
        "benchmark_result": str(result_path),
        "suite_name": result["suite_name"],
        "run_count": result["run_count"],
        "error_count": result["error_count"],
        "registry": result.get("registry_path"),
    }, indent=2, sort_keys=True))
    return 0 if result["error_count"] == 0 else 1


def cmd_dissect(args) -> int:
    raw = read_table(args.path)
    raw = _maybe_klines(raw, getattr(args, "klines_format", False))
    raw = _slice_rows(raw, getattr(args, "skip_rows", None), args.max_rows)
    feature_frame = build_feature_frame(
        raw,
        feature_mode=args.feature_mode,
        columns=_columns(args.columns),
        close_col=args.close_col,
        umap_train_size=args.umap_train_size,
        smooth_windows=tuple(args.smooth_windows),
    )
    if args.reference_path:
        reference = read_table(args.reference_path)
        reference = _maybe_klines(reference, args.reference_klines_format)
        feature_frame = append_reference_context(
            feature_frame,
            raw,
            reference,
            timestamp_col=args.timestamp_col,
            close_col=args.close_col,
            windows=tuple(args.relative_windows),
            tolerance=args.reference_tolerance,
        )
    asset_id = args.asset_id or Path(args.path).stem
    dissected = dissect_feature_frame(
        feature_frame,
        asset_id=asset_id,
        horizons=tuple(args.horizons),
        train_size=args.train_size,
        train_fraction=args.train_fraction,
        phase_window=args.phase_window,
    )
    artifacts = write_asset_dissect_bundle(dissected, args.output)
    result = dissected["result"]
    print(json.dumps({
        "asset_id": asset_id,
        "artifacts": artifacts,
        "rows": result["rows"],
        "horizons": result["horizons"],
        "top_correlations": result["top_feature_target_correlations"][:5],
    }, indent=2, sort_keys=True))
    return 0


def cmd_dissect_many(args) -> int:
    result = run_universe_dissect(args.universe, args.output)
    result_path = Path(args.output) / "universe_dissect.json"
    write_universe_dissect_result(result, result_path)
    print(json.dumps({
        "universe_dissect": str(result_path),
        "name": result["name"],
        "asset_count": result["asset_count"],
        "error_count": result["error_count"],
        "tables": result["tables"],
    }, indent=2, sort_keys=True))
    return 0 if result["error_count"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eso.cli", description="ESO agent-ready explorer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Inspect a dataset without running ESO")
    p.add_argument("path")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--output")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("diagnose", help="Run blind diagnostics only")
    p.add_argument("path")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--normalize", default="robust", choices=["none", "robust", "standard", "minmax"])
    p.add_argument("--skip-rows", type=int, help="Skip first N rows (for temporal windowing)")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true",
                   help="Normalise Binance Klines column names before feature engineering")
    p.add_argument("--window-size", type=int)
    p.add_argument("--window-step", type=int, default=1)
    p.add_argument("--window-mode", default="last", choices=["last", "mean", "flat"])
    p.add_argument("--feature-mode", default="raw", choices=_FEATURE_MODES,
                   help="Transform input to financial features before diagnosing")
    p.add_argument("--output")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("explore", help="Run full ESO exploration and write a report bundle")
    p.add_argument("path")
    p.add_argument("--dataset-id")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--normalize", default="robust", choices=["none", "robust", "standard", "minmax"])
    p.add_argument("--skip-rows", type=int, help="Skip first N rows (for temporal windowing)")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true",
                   help="Normalise Binance Klines column names before feature engineering")
    p.add_argument("--window-size", type=int)
    p.add_argument("--window-step", type=int, default=1)
    p.add_argument("--window-mode", default="last", choices=["last", "mean", "flat"])
    p.add_argument("--feature-mode", default="raw", choices=_FEATURE_MODES,
                   help="Transform input to financial features before exploring")
    p.add_argument("--manifolds", nargs="+", default=["circle", "sphere2", "torus2", "cylinder"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--n-masks", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--projection-method", default="linear",
                   choices=["linear", "umap", "isomap"],
                   help="Embedding method before manifold projection (linear=SVD, umap, isomap)")
    p.add_argument("--proj-neighbors", type=int, default=15,
                   help="n_neighbors for UMAP/Isomap projection")
    p.add_argument("--output", default="reports/eso_run")
    p.add_argument("--registry", default="experiments/eso_registry.csv")
    p.add_argument("--no-registry", action="store_true")
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("model-eval", help="Run a supervised temporal model baseline and write a manifest")
    p.add_argument("path")
    p.add_argument("--dataset-id")
    p.add_argument("--task", required=True, choices=["volatility", "regime", "direction"])
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--columns", nargs="+", help="Input columns for raw mode or engineered feature columns for feature modes")
    p.add_argument("--feature-columns", nargs="+", help="Final model feature columns; defaults to task-aware columns")
    p.add_argument("--feature-mode", default="raw", choices=_MODEL_FEATURE_MODES)
    p.add_argument("--close-col", default="close")
    p.add_argument("--umap-train-size", type=int, default=27000)
    p.add_argument("--smooth-windows", type=int, nargs="+", default=[6, 24, 72])
    p.add_argument("--skip-rows", type=int)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true",
                   help="Normalise Binance Klines column names before feature engineering")
    p.add_argument("--train-size", type=int)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--model-type", default="auto", choices=["auto", "ridge", "logistic", "rf"])
    p.add_argument("--output", default="reports/model_run")
    p.add_argument("--registry", default="reports/model_runs.csv")
    p.add_argument("--no-registry", action="store_true")
    p.set_defaults(func=cmd_model_eval)

    p = sub.add_parser("model-runs", help="List or compare supervised model-run manifests")
    p.add_argument("action", choices=["list", "compare"])
    p.add_argument("refs", nargs="*", help="Run ids or manifest paths for compare")
    p.add_argument("--registry", default="reports/model_runs.csv")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--task", choices=["volatility", "regime", "direction"])
    p.add_argument("--dataset-id")
    p.set_defaults(func=cmd_model_runs)

    p = sub.add_parser("validate-json", help="Validate an ESO JSON artifact against its schema_version")
    p.add_argument("path")
    p.add_argument("--schema-version")
    p.set_defaults(func=cmd_validate_json)

    p = sub.add_parser("benchmark", help="Run a declarative benchmark suite JSON")
    p.add_argument("suite")
    p.add_argument("--output", default="reports/benchmark_run")
    p.add_argument("--registry", default="reports/model_runs.csv")
    p.add_argument("--no-registry", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("dissect", help="Dissect one market asset into targets, correlations, regimes, and phase sectors")
    p.add_argument("path")
    p.add_argument("--asset-id")
    p.add_argument("--feature-mode", default="compact", choices=_MODEL_FEATURE_MODES)
    p.add_argument("--columns", nargs="+")
    p.add_argument("--close-col", default="close")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--horizons", type=int, nargs="+", default=[3, 12])
    p.add_argument("--phase-window", type=int, default=24)
    p.add_argument("--train-size", type=int)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--skip-rows", type=int)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true")
    p.add_argument("--umap-train-size", type=int, default=27000)
    p.add_argument("--smooth-windows", type=int, nargs="+", default=[6, 24, 72])
    p.add_argument("--reference-path", help="Optional reference asset, e.g. BTC, for relative features")
    p.add_argument("--reference-klines-format", action="store_true")
    p.add_argument("--reference-tolerance", help="Optional merge_asof tolerance, e.g. 4h")
    p.add_argument("--relative-windows", type=int, nargs="+", default=[24, 72])
    p.add_argument("--output", default="reports/asset_dissect")
    p.set_defaults(func=cmd_dissect)

    p = sub.add_parser("dissect-many", help="Run asset dissection for a declarative universe JSON")
    p.add_argument("universe")
    p.add_argument("--output", default="reports/universe_dissect")
    p.set_defaults(func=cmd_dissect_many)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ESO error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
