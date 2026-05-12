"""Command line interface for ESO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eso.data.loader import load_dataset, read_table
from eso.data.validation import validate_dataframe
from eso.diagnostics.report import run_diagnosis
from eso.pipeline import ESOExplorer
from eso.reporting import write_report_bundle


def _columns(values):
    return values if values else None


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


def cmd_diagnose(args) -> int:
    loaded = load_dataset(
        args.path,
        columns=_columns(args.columns),
        normalize_method=args.normalize,
        max_rows=args.max_rows,
        window_size=args.window_size,
        window_step=args.window_step,
        window_mode=args.window_mode,
    )
    report = run_diagnosis(loaded.data)
    out = {"dataset": loaded.info(), "diagnosis": report}
    print(report.get("summary", "diagnosis complete"))
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def cmd_explore(args) -> int:
    explorer = ESOExplorer(registry_path=args.registry)
    loaded = load_dataset(
        args.path,
        columns=_columns(args.columns),
        normalize_method=args.normalize,
        max_rows=args.max_rows,
        window_size=args.window_size,
        window_step=args.window_step,
        window_mode=args.window_mode,
    )
    dataset_id = args.dataset_id or Path(args.path).stem
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
    )
    report["dataset"] = loaded.info()
    artifacts = write_report_bundle(report, loaded.data, args.output)
    print(json.dumps({"best": report.get("best"), "artifacts": artifacts}, indent=2, sort_keys=True))
    return 0


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
    p.add_argument("--max-rows", type=int)
    p.add_argument("--window-size", type=int)
    p.add_argument("--window-step", type=int, default=1)
    p.add_argument("--window-mode", default="last", choices=["last", "mean", "flat"])
    p.add_argument("--output")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("explore", help="Run full ESO exploration and write a report bundle")
    p.add_argument("path")
    p.add_argument("--dataset-id")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--normalize", default="robust", choices=["none", "robust", "standard", "minmax"])
    p.add_argument("--max-rows", type=int)
    p.add_argument("--window-size", type=int)
    p.add_argument("--window-step", type=int, default=1)
    p.add_argument("--window-mode", default="last", choices=["last", "mean", "flat"])
    p.add_argument("--manifolds", nargs="+", default=["circle", "sphere2", "torus2", "cylinder"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.25)
    p.add_argument("--n-masks", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="reports/eso_run")
    p.add_argument("--registry", default="experiments/eso_registry.csv")
    p.add_argument("--no-registry", action="store_true")
    p.set_defaults(func=cmd_explore)

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
