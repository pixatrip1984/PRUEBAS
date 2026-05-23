"""Rigorous validation pass for a single asset.

Three orthogonal tests, all on top of build_alt_feature_vector:

  1. Half-split: cut the OOS window in two halves, report net Sharpe and
     return per half. A real signal should be positive in both; an
     in-sample artefact will have most of the gain in one half.

  2. Parameter sweep: vary gate_percentile × signal_col × enter_threshold
     and report whether the result is robust or fragile.

  3. Stability: rolling correlation of cos_theta_24h vs future return,
     count sign flips, cross-correlate with ring_radius and vol_20.

Output: validation_<asset>.json + validation_<asset>.md per asset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from eso.backtest import BacktestConfig, gated_phase_threshold_strategy, run_backtest
from eso.diagnostics.signal_stability import analyse_signal_stability
from eso.lab.multi_asset_sweep import build_alt_feature_vector


@dataclass
class HalfSplitMetrics:
    n_bars: int
    trades: int
    gross_return: float
    net_return: float
    gross_sharpe: float
    net_sharpe: float
    hit_rate: float


@dataclass
class SweepEntry:
    gate_percentile: float
    signal_col: str
    enter_threshold: float
    trades: int
    gross_sharpe: float
    net_sharpe: float
    net_return: float
    edge_bps_per_trade: float


@dataclass
class ValidationReport:
    asset: str
    n_bars: int
    half_split: dict          # {"first_half": HalfSplitMetrics, "second_half": HalfSplitMetrics}
    half_split_passed: bool   # both halves positive net return
    sweep: list               # list of SweepEntry as dicts
    sweep_robust: bool        # at least 50% of param combos net-positive
    sweep_best_net_sharpe: float
    stability_mean_corrs: dict
    stability_flip_counts: dict
    stability_cross_corr: dict
    verdict: str


def _half_split(prices: pd.Series, positions: pd.Series, cfg: BacktestConfig) -> dict:
    """Compute per-half metrics by splitting prices/positions in two."""
    n = len(prices)
    mid = n // 2
    out = {}
    for name, sl in [("first_half", slice(0, mid)), ("second_half", slice(mid, n))]:
        p_slice = prices.iloc[sl]
        pos_slice = positions.iloc[sl]
        # Reset positions at boundary: assume flat at start of each half for fair compare
        pos_reset = pos_slice.copy()
        pos_reset.iloc[0] = 0.0
        if len(p_slice) < 10:
            continue
        bt = run_backtest(p_slice, pos_reset, cfg)
        out[name] = asdict(HalfSplitMetrics(
            n_bars=len(p_slice),
            trades=bt.metrics["n_trades"],
            gross_return=bt.metrics["gross_total_return"],
            net_return=bt.metrics["net_total_return"],
            gross_sharpe=bt.metrics["gross_sharpe"],
            net_sharpe=bt.metrics["net_sharpe"],
            hit_rate=bt.metrics["hit_rate"],
        ))
    return out


def _parameter_sweep(
    fv: pd.DataFrame,
    cfg: BacktestConfig,
    gate_percentiles: tuple[float, ...] = (0.30, 0.50, 0.70, 0.85),
    signal_cols: tuple[str, ...] = ("cos_theta_6h", "cos_theta_24h", "cos_theta_72h",
                                    "sin_theta_24h"),
    enter_thresholds: tuple[float, ...] = (0.2, 0.3, 0.4),
) -> list[dict]:
    rows = []
    prices = fv["close"]
    for gp in gate_percentiles:
        for sc in signal_cols:
            if sc not in fv.columns:
                continue
            for et in enter_thresholds:
                pos = gated_phase_threshold_strategy(
                    fv, signal_col=sc, gate_col="ring_radius",
                    gate_percentile=gp, gate_window=min(500, len(fv) // 3),
                    enter_threshold=et, exit_threshold=et / 3,
                    allow_short=True,
                )
                bt = run_backtest(prices, pos, cfg)
                n_tr = bt.metrics["n_trades"]
                edge = (bt.metrics["gross_total_return"] * 1e4 / n_tr) if n_tr > 0 else 0.0
                rows.append(asdict(SweepEntry(
                    gate_percentile=gp, signal_col=sc, enter_threshold=et,
                    trades=n_tr,
                    gross_sharpe=bt.metrics["gross_sharpe"],
                    net_sharpe=bt.metrics["net_sharpe"],
                    net_return=bt.metrics["net_total_return"],
                    edge_bps_per_trade=edge,
                )))
    return rows


def validate_asset(
    csv_path: str | Path,
    output_dir: str | Path,
    bars_per_year: int = 2190,
    fee_bps: float = 5.5,
    slippage_bps: float = 2.0,
    seed: int = 42,
) -> ValidationReport:
    """Run the three-test validation on one asset and write a report."""
    csv_path = Path(csv_path)
    asset = csv_path.stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    fv = build_alt_feature_vector(df, seed=seed)

    cfg = BacktestConfig(
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        bars_per_year=bars_per_year, allow_short=True,
    )

    # Reference run with the sweep-winning parameters
    pos_ref = gated_phase_threshold_strategy(
        fv, signal_col="cos_theta_24h", gate_col="ring_radius",
        gate_percentile=0.50, gate_window=min(500, len(fv) // 3),
        enter_threshold=0.3, exit_threshold=0.1,
        allow_short=True,
    )

    # (1) Half-split
    halves = _half_split(fv["close"], pos_ref, cfg)
    half_passed = (
        halves.get("first_half", {}).get("net_return", -1) > 0
        and halves.get("second_half", {}).get("net_return", -1) > 0
    )

    # (2) Parameter sweep
    sweep = _parameter_sweep(fv, cfg)
    if sweep:
        n_positive = sum(1 for r in sweep if r["net_return"] > 0)
        sweep_robust = n_positive / len(sweep) >= 0.5
        best_sharpe = max(r["net_sharpe"] for r in sweep)
    else:
        sweep_robust = False
        best_sharpe = float("nan")

    # (3) Stability
    stab = analyse_signal_stability(
        fv, horizon=12, window=min(500, len(fv) // 3),
    )

    # Verdict
    if half_passed and sweep_robust and best_sharpe >= 1.0:
        verdict = "STRONG — survives temporal split AND parameter robustness"
    elif half_passed and sweep_robust:
        verdict = "ROBUST — both halves positive and parameter-stable"
    elif half_passed:
        verdict = "FRAGILE — temporal split OK but parameter-sensitive"
    elif sweep_robust:
        verdict = "REGIME-DEPENDENT — parameter-stable but not temporally stable"
    else:
        verdict = "REJECTED — single-period in-sample artefact"

    report = ValidationReport(
        asset=asset,
        n_bars=len(df),
        half_split=halves,
        half_split_passed=half_passed,
        sweep=sweep,
        sweep_robust=sweep_robust,
        sweep_best_net_sharpe=best_sharpe,
        stability_mean_corrs=stab.mean_corr,
        stability_flip_counts=stab.flip_count,
        stability_cross_corr=stab.cross_corr_with_regime,
        verdict=verdict,
    )

    # Write JSON
    (out / f"validation_{asset}.json").write_text(
        json.dumps(asdict(report), indent=2, default=float),
        encoding="utf-8",
    )

    # Write markdown
    md = [f"# Validation: {asset}", "", f"**Verdict: {verdict}**", ""]
    md.append(f"Bars: {report.n_bars}    Feature-vector rows: {len(fv)}")
    md.append("")
    md.append("## 1. Half-split (temporal robustness)")
    md.append("")
    md.append("| Half | Bars | Trades | Gross Sharpe | Net Sharpe | Net Return |")
    md.append("|------|-----:|-------:|-------------:|-----------:|-----------:|")
    for name in ("first_half", "second_half"):
        h = halves.get(name)
        if not h:
            continue
        md.append(
            f"| {name} | {h['n_bars']} | {h['trades']} | "
            f"{h['gross_sharpe']:+.2f} | {h['net_sharpe']:+.2f} | "
            f"{h['net_return']*100:+.1f}% |"
        )
    md.append(f"\n**Passed half-split:** {'YES' if half_passed else 'NO'}")
    md.append("")
    md.append("## 2. Parameter sweep (robustness)")
    md.append("")
    md.append(f"Tested {len(sweep)} combinations of (gate × signal × enter_threshold).")
    md.append(f"- Positive net return: {sum(1 for r in sweep if r['net_return']>0)} / {len(sweep)}")
    md.append(f"- Best net Sharpe: {best_sharpe:.2f}")
    md.append(f"- Robust (>=50% positive): {'YES' if sweep_robust else 'NO'}")
    md.append("")
    md.append("Top 5 by net Sharpe:")
    md.append("")
    md.append("| Gate | Signal | Enter | Trades | GrossSh | NetSh | NetRet | Edge |")
    md.append("|-----:|--------|------:|-------:|--------:|------:|-------:|-----:|")
    top = sorted(sweep, key=lambda r: -r["net_sharpe"])[:5]
    for r in top:
        md.append(
            f"| {r['gate_percentile']:.2f} | {r['signal_col']} | {r['enter_threshold']:.2f} | "
            f"{r['trades']} | {r['gross_sharpe']:+.2f} | {r['net_sharpe']:+.2f} | "
            f"{r['net_return']*100:+.1f}% | {r['edge_bps_per_trade']:+.0f}bps |"
        )
    md.append("")
    md.append("## 3. Signal stability")
    md.append("")
    md.append("| Feature | mean rolling corr | sign flips |")
    md.append("|---------|------------------:|-----------:|")
    for c in stab.mean_corr:
        md.append(f"| {c} | {stab.mean_corr[c]:+.3f} | {stab.flip_count[c]} |")

    (out / f"validation_{asset}.md").write_text("\n".join(md), encoding="utf-8")
    return report
