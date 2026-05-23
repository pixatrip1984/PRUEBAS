"""Walk-forward parameter selection for cycle-phase strategies.

The validation showed that the signal is parameter-stable but not
temporally stable. Walk-forward is the canonical response:

  Fold 0 (warmup):  no trading. Used only to seed parameter history.
  Fold k (k >= 1):  pick best (gate, signal_col, enter_threshold) by
                    running a sweep over the *concatenated previous folds*,
                    then trade fold k with that choice. No lookahead.

The equity curve is the concatenation of fold-by-fold backtests with
positions reset to flat at each fold boundary (conservative — pays an
extra entry cost per fold, but eliminates carry-over assumptions).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from eso.backtest import BacktestConfig, gated_phase_threshold_strategy, run_backtest
from eso.lab.multi_asset_sweep import build_alt_feature_vector


SWEEP_GRID = [
    (gp, sc, et)
    for gp in (0.30, 0.50, 0.70, 0.85)
    for sc in ("cos_theta_6h", "cos_theta_24h", "cos_theta_72h", "sin_theta_24h")
    for et in (0.2, 0.3, 0.4)
]  # 48 combos


@dataclass
class FoldRecord:
    fold: int
    bars: tuple[int, int]
    selected_gate: float
    selected_signal: str
    selected_enter: float
    selection_metric: float       # net Sharpe on selection window
    trades: int
    gross_return: float
    net_return: float
    gross_sharpe: float
    net_sharpe: float


@dataclass
class WalkForwardResult:
    asset: str
    n_bars: int
    n_folds: int
    folds: list                   # list of FoldRecord dicts (skipping warmup fold 0)
    aggregate_net_return: float
    aggregate_gross_sharpe: float
    aggregate_net_sharpe: float
    aggregate_trades: int
    verdict: str


def _select_best_params(
    fv_slice: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[float, str, float, float]:
    """Sweep the grid on a slice and return the (gate, signal, enter, score) of best."""
    best = (None, float("-inf"))
    prices = fv_slice["close"]
    for gp, sc, et in SWEEP_GRID:
        if sc not in fv_slice.columns:
            continue
        pos = gated_phase_threshold_strategy(
            fv_slice, signal_col=sc, gate_col="ring_radius",
            gate_percentile=gp, gate_window=min(300, len(fv_slice) // 3),
            enter_threshold=et, exit_threshold=et / 3,
            allow_short=True,
        )
        bt = run_backtest(prices, pos, cfg)
        score = bt.metrics["net_sharpe"]
        if not np.isnan(score) and score > best[1]:
            best = ((gp, sc, et), score)
    gp, sc, et = best[0]
    return gp, sc, et, best[1]


def walk_forward_backtest(
    csv_path: str | Path,
    n_folds: int = 4,
    output_dir: str | Path | None = None,
    bars_per_year: int = 2190,
    fee_bps: float = 5.5,
    slippage_bps: float = 2.0,
    seed: int = 42,
) -> WalkForwardResult:
    """Walk-forward parameter selection + backtest on one asset.

    fold 0 is warmup (no trading); folds 1..n_folds-1 are traded with
    params selected from the concatenated history of prior folds.
    """
    csv_path = Path(csv_path)
    asset = csv_path.stem
    df = pd.read_csv(csv_path)
    fv = build_alt_feature_vector(df, seed=seed)
    if len(fv) < 400:
        raise ValueError(f"Feature vector too short ({len(fv)}) for walk-forward")

    cfg = BacktestConfig(
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        bars_per_year=bars_per_year, allow_short=True,
    )

    fold_size = len(fv) // n_folds
    fold_edges = [(i * fold_size, (i + 1) * fold_size) for i in range(n_folds)]
    fold_edges[-1] = (fold_edges[-1][0], len(fv))  # absorb remainder into last fold

    records = []
    gross_log_pnl = 0.0
    net_log_pnl = 0.0
    total_trades = 0
    net_log_returns_all = []

    for k in range(1, n_folds):
        # Selection window = concatenation of fold 0 .. k-1
        sel_start = fold_edges[0][0]
        sel_end = fold_edges[k - 1][1]
        fv_sel = fv.iloc[sel_start:sel_end].copy()
        if len(fv_sel) < 100:
            continue

        gp, sc, et, score = _select_best_params(fv_sel, cfg)

        # Trade window
        tr_start, tr_end = fold_edges[k]
        fv_tr = fv.iloc[tr_start:tr_end].copy()
        if len(fv_tr) < 30:
            continue

        pos = gated_phase_threshold_strategy(
            fv_tr, signal_col=sc, gate_col="ring_radius",
            gate_percentile=gp, gate_window=min(300, len(fv_tr) // 3),
            enter_threshold=et, exit_threshold=et / 3,
            allow_short=True,
        )
        bt = run_backtest(fv_tr["close"], pos, cfg)

        gross_log_pnl += float(np.log1p(bt.metrics["gross_total_return"]))
        net_log_pnl += float(np.log1p(bt.metrics["net_total_return"]))
        total_trades += bt.metrics["n_trades"]
        net_log_returns_all.extend(bt.net_returns.tolist())

        records.append(asdict(FoldRecord(
            fold=k,
            bars=(tr_start, tr_end),
            selected_gate=gp,
            selected_signal=sc,
            selected_enter=et,
            selection_metric=score,
            trades=bt.metrics["n_trades"],
            gross_return=bt.metrics["gross_total_return"],
            net_return=bt.metrics["net_total_return"],
            gross_sharpe=bt.metrics["gross_sharpe"],
            net_sharpe=bt.metrics["net_sharpe"],
        )))

    agg_gross = float(np.expm1(gross_log_pnl))
    agg_net = float(np.expm1(net_log_pnl))

    # Aggregate Sharpe from concatenated net returns
    net_arr = np.array(net_log_returns_all, dtype=float)
    gross_sharpe_agg = float("nan")
    net_sharpe_agg = float("nan")
    if len(net_arr) > 1 and net_arr.std(ddof=0) > 0:
        net_sharpe_agg = float(net_arr.mean() / net_arr.std(ddof=0) * np.sqrt(bars_per_year))

    # Verdict
    n_winning_folds = sum(1 for r in records if r["net_return"] > 0)
    n_traded = len(records)
    if n_traded == 0:
        verdict = "INCONCLUSIVE — no folds traded"
    elif agg_net > 0 and net_sharpe_agg >= 1.0:
        verdict = "TRADEABLE — positive aggregate AND net Sharpe >= 1.0"
    elif agg_net > 0 and n_winning_folds / n_traded >= 0.6:
        verdict = "PROMISING — positive aggregate and majority of folds positive"
    elif agg_net > 0:
        verdict = "MARGINAL — positive aggregate but few winning folds"
    else:
        verdict = "REJECTED — negative aggregate net return"

    result = WalkForwardResult(
        asset=asset,
        n_bars=len(df),
        n_folds=n_folds,
        folds=records,
        aggregate_net_return=agg_net,
        aggregate_gross_sharpe=gross_sharpe_agg,
        aggregate_net_sharpe=net_sharpe_agg,
        aggregate_trades=total_trades,
        verdict=verdict,
    )

    if output_dir is not None:
        _write_report(result, Path(output_dir))
    return result


def _write_report(result: WalkForwardResult, out_dir: Path) -> None:
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"walkforward_{result.asset}.json").write_text(
        json.dumps(asdict(result), indent=2, default=float),
        encoding="utf-8",
    )
    lines = [
        f"# Walk-forward validation: {result.asset}",
        "",
        f"**Verdict: {result.verdict}**",
        "",
        f"Folds traded: {len(result.folds)} / {result.n_folds - 1}",
        f"Aggregate net return: {result.aggregate_net_return*100:+.1f}%",
        f"Aggregate net Sharpe: {result.aggregate_net_sharpe:+.2f}",
        f"Aggregate trades: {result.aggregate_trades}",
        "",
        "## Per-fold breakdown",
        "",
        "| Fold | Bars | Selected (gate/sig/enter) | Trades | NetSh | NetRet |",
        "|-----:|------|----------------------------|-------:|------:|-------:|",
    ]
    for r in result.folds:
        lines.append(
            f"| {r['fold']} | {r['bars'][0]}–{r['bars'][1]} | "
            f"p{int(r['selected_gate']*100)}/{r['selected_signal']}/{r['selected_enter']:.2f} | "
            f"{r['trades']} | {r['net_sharpe']:+.2f} | {r['net_return']*100:+.1f}% |"
        )
    (out_dir / f"walkforward_{result.asset}.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def walk_forward_many(
    csv_paths: Sequence[str | Path],
    output_dir: str | Path = "reports/walkforward",
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for path in csv_paths:
        asset = Path(path).stem
        print(f"-> {asset} walk-forward ...", flush=True)
        try:
            r = walk_forward_backtest(path, output_dir=output_dir, **kwargs)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        print(f"  verdict={r.verdict}   net={r.aggregate_net_return*100:+.1f}%   "
              f"sharpe={r.aggregate_net_sharpe:+.2f}")
        rows.append({
            "asset": r.asset,
            "verdict": r.verdict,
            "aggregate_net_return": r.aggregate_net_return,
            "aggregate_net_sharpe": r.aggregate_net_sharpe,
            "aggregate_trades": r.aggregate_trades,
            "folds_traded": len(r.folds),
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("aggregate_net_sharpe", ascending=False)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "walkforward_ranking.csv", index=False)
        return df
    return pd.DataFrame()
