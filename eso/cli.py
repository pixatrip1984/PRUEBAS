"""Command line interface for ESO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eso.data.features import build_financial_features, feature_column_groups, prepare_btc_klines
from eso.data.loader import load_dataset, read_table
from eso.data.validation import validate_dataframe
from eso.diagnostics.report import run_diagnosis
from eso.pipeline import ESOExplorer
from eso.reporting import write_report_bundle


def cmd_stability(args) -> int:
    """Analyse rolling stability of cycle-phase predictive power."""
    from eso.signals.feature_vector import build_feature_vector, CausalFeatureBuilder
    from eso.diagnostics.signal_stability import analyse_signal_stability
    from eso.diagnostics.stability_report import write_stability_report

    df = read_table(args.path)
    df = _maybe_klines(df, getattr(args, "klines_format", False))
    df = _slice_rows(df, getattr(args, "skip_rows", None), args.max_rows)

    if args.causal_mode == "rolling":
        builder = CausalFeatureBuilder(
            init_train=args.init_train,
            refit_every=args.refit_every,
            seed=args.seed,
        )
        fv = builder.fit_predict(df)
    else:
        fv = build_feature_vector(df, train_size=args.train_size, seed=args.seed)

    if fv.empty:
        print("ESO error: empty feature vector", file=sys.stderr)
        return 2

    report = analyse_signal_stability(
        fv,
        horizon=args.horizon,
        window=args.window,
    )
    print(report.summary)
    if args.output:
        artifacts = write_stability_report(
            report, args.output,
            title=f"{Path(args.path).stem} — stability h={args.horizon} w={args.window}",
        )
        print(json.dumps({"artifacts": artifacts}, indent=2))
    return 0


def cmd_backtest(args) -> int:
    """Run a cost-aware backtest of a causal phase strategy on a price series."""
    import pandas as pd
    from eso.signals.feature_vector import build_feature_vector, CausalFeatureBuilder
    from eso.backtest import (
        BacktestConfig,
        run_backtest,
        phase_threshold_strategy,
        proportional_strategy,
        proportional_deadband_strategy,
        regime_gated_strategy,
        gated_phase_threshold_strategy,
        long_only_baseline,
        model_strategy,
    )
    from eso.backtest.report import write_backtest_report

    df = read_table(args.path)
    df = _maybe_klines(df, getattr(args, "klines_format", False))
    df = _slice_rows(df, getattr(args, "skip_rows", None), args.max_rows)

    if args.causal_mode == "rolling":
        builder = CausalFeatureBuilder(
            init_train=args.init_train,
            refit_every=args.refit_every,
            seed=args.seed,
        )
        fv = builder.fit_predict(df)
    else:
        fv = build_feature_vector(df, train_size=args.train_size, seed=args.seed)

    if fv.empty:
        print("ESO error: empty feature vector — check data length", file=sys.stderr)
        return 2

    if args.strategy == "long_only":
        pos = long_only_baseline(fv)
    elif args.strategy == "proportional":
        pos = proportional_strategy(
            fv,
            signal_col=args.signal_col,
            confidence_col=args.confidence_col,
            signal_scale=args.signal_scale,
        )
    elif args.strategy == "proportional_deadband":
        pos = proportional_deadband_strategy(
            fv,
            signal_col=args.signal_col,
            confidence_col=args.confidence_col,
            signal_scale=args.signal_scale,
            min_trade_size=args.min_trade_size,
        )
    elif args.strategy == "gated_phase_threshold":
        pos = gated_phase_threshold_strategy(
            fv,
            signal_col=args.signal_col,
            gate_col=args.gate_col,
            gate_percentile=args.gate_percentile,
            gate_window=args.gate_window,
            enter_threshold=args.enter_threshold,
            exit_threshold=args.exit_threshold,
            allow_short=not args.long_only,
        )
    elif args.strategy == "regime_gated":
        pos = regime_gated_strategy(
            fv,
            signal_col=args.signal_col,
            gate_col=args.gate_col,
            gate_percentile=args.gate_percentile,
            gate_window=args.gate_window,
            signal_scale=args.signal_scale,
            min_trade_size=args.min_trade_size,
            allow_short=not args.long_only,
        )
    elif args.strategy == "model":
        result = model_strategy(
            fv,
            horizon=args.model_horizon,
            train_frac=args.model_train_frac,
            ridge_alpha=args.ridge_alpha,
            signal_scale=args.signal_scale,
            min_trade_size=args.min_trade_size,
            return_diagnostics=True,
        )
        pos = result.positions
        print(f"Model train R²: {result.train_score:+.4f}   "
              f"test R²: {result.test_score:+.4f}   "
              f"horizon: {result.target_horizon}h")
        top_coefs = sorted(result.coefficients.items(), key=lambda kv: -abs(kv[1]))[:5]
        print("Top |coef|: " + ", ".join(f"{c}={w:+.4f}" for c, w in top_coefs))
    else:
        pos = phase_threshold_strategy(
            fv,
            signal_col=args.signal_col,
            enter_threshold=args.enter_threshold,
            exit_threshold=args.exit_threshold,
            allow_short=not args.long_only,
        )

    cfg = BacktestConfig(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        bars_per_year=args.bars_per_year,
        allow_short=not args.long_only,
    )

    prices = fv["close"]
    result = run_backtest(prices, pos, cfg)

    print(result.summary())

    if args.output:
        artifacts = write_backtest_report(
            result,
            args.output,
            title=f"{Path(args.path).stem} — {args.strategy}",
        )
        print(json.dumps({"artifacts": artifacts}, indent=2))

    return 0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eso.cli", description="ESO agent-ready explorer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Inspect a dataset without running ESO")
    p.add_argument("path")
    p.add_argument("--columns", nargs="+")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--output")
    p.set_defaults(func=cmd_inspect)

    _FEATURE_MODES = ["raw", "returns_only", "returns_vol", "microstructure", "compact", "full"]

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

    p = sub.add_parser("backtest", help="Cost-aware backtest of a causal phase strategy")
    p.add_argument("path")
    p.add_argument("--skip-rows", type=int)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true")
    p.add_argument("--causal-mode", choices=["split", "rolling"], default="split",
                   help="split = fit UMAP once on train window; rolling = refit periodically")
    p.add_argument("--train-size", type=int, default=27000,
                   help="(split mode) bars used to fit the UMAP")
    p.add_argument("--init-train", type=int, default=10000,
                   help="(rolling mode) initial training window")
    p.add_argument("--refit-every", type=int, default=2000,
                   help="(rolling mode) refit cadence in bars")
    p.add_argument("--strategy",
                   choices=["phase_threshold", "proportional", "proportional_deadband",
                            "regime_gated", "gated_phase_threshold",
                            "model", "long_only"],
                   default="phase_threshold")
    p.add_argument("--confidence-col", default="ring_radius",
                   help="(proportional) feature column used as position-size weight")
    p.add_argument("--signal-scale", type=float, default=1.0,
                   help="(proportional) scalar multiplier on raw signal before clipping")
    p.add_argument("--min-trade-size", type=float, default=0.15,
                   help="(proportional_deadband / model) minimum |Δposition| to trigger a rebalance")
    p.add_argument("--model-horizon", type=int, default=12,
                   help="(model) target prediction horizon in bars")
    p.add_argument("--model-train-frac", type=float, default=0.5,
                   help="(model) fraction of feature vector used to fit the regression")
    p.add_argument("--ridge-alpha", type=float, default=1.0,
                   help="(model) Ridge L2 regularisation strength")
    p.add_argument("--gate-col", default="ring_radius",
                   help="(regime_gated) regime indicator column")
    p.add_argument("--gate-percentile", type=float, default=0.70,
                   help="(regime_gated) trade only when gate > this rolling percentile")
    p.add_argument("--gate-window", type=int, default=1000,
                   help="(regime_gated) rolling window for the gate percentile")
    p.add_argument("--signal-col", default="cos_theta_24h",
                   help="Feature column to drive entries/exits")
    p.add_argument("--enter-threshold", type=float, default=0.3)
    p.add_argument("--exit-threshold", type=float, default=0.1)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--fee-bps", type=float, default=5.5,
                   help="Per-side fee in bps (Bybit perp taker default)")
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--bars-per-year", type=int, default=24 * 365,
                   help="Annualisation factor (8760 for 1h bars, 2190 for 4h)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", help="Directory for report artifacts")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("stability",
                       help="Rolling stability analysis of cycle-phase predictive power")
    p.add_argument("path")
    p.add_argument("--skip-rows", type=int)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--klines-format", action="store_true")
    p.add_argument("--causal-mode", choices=["split", "rolling"], default="split")
    p.add_argument("--train-size", type=int, default=27000)
    p.add_argument("--init-train", type=int, default=10000)
    p.add_argument("--refit-every", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--window", type=int, default=1000,
                   help="Rolling window in bars for correlation estimation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", help="Directory for report artifacts")
    p.set_defaults(func=cmd_stability)

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
