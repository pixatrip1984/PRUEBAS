"""Paper-trading journal — simulate live_signal day-by-day on history.

This is the final pre-deployment validation: replay the live_signal
pipeline at each historical step, record the recommended portfolio
position, then compute realized P&L when the next bar arrives.

The result is a JSON + CSV journal you can review bar-by-bar, identical
in logic to what live deployment would have done at that timestamp.

Why this matters:
  - Backtests can be subtly different from live (lookahead in feature
    construction, off-by-one in selection windows, etc.).
  - The journal uses the SAME functions as live_signal — so if its
    aggregated Sharpe matches the walk-forward portfolio Sharpe, we
    have evidence the live code is faithful.
  - The journal also tells you what your worst day would have looked
    like, what your typical position size is, how often you actually
    trade. Operational intuition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from eso.backtest import BacktestConfig, gated_phase_threshold_strategy, run_backtest
from eso.lab.live_signal import (
    LIVE_ASSET_CONFIGS,
    asset_live_signal,
    s_bars_per_year,
)
from eso.lab.multi_asset_sweep import build_alt_feature_vector
from eso.lab.walk_forward import SWEEP_GRID


@dataclass
class JournalEntry:
    timestamp: str
    asset: str
    target_position: float
    held_position: float
    delta: float
    realized_log_return: float
    selection_sharpe: float
    quality_pass: bool


@dataclass
class JournalSummary:
    asset_count: int
    n_steps: int
    aggregate_log_return: float
    aggregate_net_return: float
    aggregate_sharpe: float
    aggregate_max_drawdown: float
    n_trades_per_asset: dict
    avg_exposure: float


def simulate_paper_trading(
    csv_path: str | Path,
    include_funding: bool = False,
    start_bar_offset: int = -300,
    selection_window_bars: int = 700,
    bars_per_year: int = 2190,
    fee_bps: float = 5.5,
    slippage_bps: float = 2.0,
    min_trade_size: float = 0.05,
    min_selection_sharpe: float = 0.5,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[JournalEntry], JournalSummary]:
    """Walk the asset CSV bar-by-bar applying live_signal at each step.

    Args:
        start_bar_offset:  Negative integer = simulate the last N bars.
                           If positive, treat as absolute starting index.
        selection_window_bars: Trailing window for parameter selection.

    Returns:
        (entries, summary). Entries is the per-bar journal.
    """
    csv_path = Path(csv_path)
    asset = csv_path.stem
    full = pd.read_csv(csv_path)
    n_full = len(full)
    if n_full < selection_window_bars + 100:
        raise ValueError(
            f"{asset}: need at least {selection_window_bars + 100} bars; "
            f"have {n_full}"
        )

    # ── FIT UMAP ONCE ────────────────────────────────────────────────────
    # build_alt_feature_vector fits the causal cycle on the first 50% of
    # the data and projects the rest. For the journal we want UMAP fitted
    # on data older than the earliest step we'll simulate, so all
    # simulated decisions are out-of-UMAP-sample.
    fv = build_alt_feature_vector(full, seed=seed, include_funding=include_funding)
    # fv covers the test slice of build_alt_feature_vector — already OOS
    # of UMAP. We will simulate decisions only over the tail of fv.

    # Adapt selection window to what's actually available
    selection_window_bars = min(selection_window_bars, max(100, len(fv) // 3))

    if start_bar_offset < 0:
        fv_start = max(selection_window_bars + 10, len(fv) + start_bar_offset)
    else:
        fv_start = start_bar_offset

    if fv_start >= len(fv) - 1:
        raise ValueError(
            f"{asset}: start offset puts the journal past end of fv "
            f"(fv_start={fv_start}, len={len(fv)})"
        )

    cfg = BacktestConfig(
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        bars_per_year=bars_per_year, allow_short=True,
    )
    cost_per_unit = cfg.cost_per_unit_turnover

    entries: list[JournalEntry] = []
    held = 0.0

    # Loop over fv rows from fv_start to end-1 (need next row for realized return)
    for end_idx in range(fv_start, len(fv) - 1):
        # Selection window: trailing slice of fv up to end_idx
        sel_start = max(0, end_idx - selection_window_bars)
        fv_sel = fv.iloc[sel_start:end_idx].copy()
        if len(fv_sel) < 100:
            continue

        # Quick sweep on selection window
        best = (None, float("-inf"))
        prices_sel = fv_sel["close"]
        for gp, sc, et in SWEEP_GRID:
            if sc not in fv_sel.columns:
                continue
            pos = gated_phase_threshold_strategy(
                fv_sel, signal_col=sc, gate_col="ring_radius",
                gate_percentile=gp,
                gate_window=min(300, len(fv_sel) // 3),
                enter_threshold=et, exit_threshold=et / 3,
                allow_short=True,
            )
            bt = run_backtest(prices_sel, pos, cfg)
            s = bt.metrics["net_sharpe"]
            if not np.isnan(s) and s > best[1]:
                best = ((gp, sc, et), s)
        if best[0] is None:
            continue
        gp, sc, et = best[0]
        sel_sharpe = best[1]

        # Apply the chosen rule to the slice up to and including end_idx,
        # take position at end_idx.
        fv_apply = fv.iloc[sel_start:end_idx + 1].copy()
        pos_series = gated_phase_threshold_strategy(
            fv_apply, signal_col=sc, gate_col="ring_radius",
            gate_percentile=gp,
            gate_window=min(300, len(fv_apply) // 3),
            enter_threshold=et, exit_threshold=et / 3,
            allow_short=True,
        )
        target = float(pos_series.iloc[-1])
        q_pass = sel_sharpe >= min_selection_sharpe

        delta = target - held
        if not q_pass:
            new_held = 0.0
        elif abs(delta) < min_trade_size:
            new_held = held
        else:
            new_held = target

        # Realized return on the NEXT fv bar
        cur_close = float(fv["close"].iloc[end_idx])
        nxt_close = float(fv["close"].iloc[end_idx + 1])
        log_ret = np.log(nxt_close / cur_close) if cur_close > 0 else 0.0

        cost = abs(new_held - held) * cost_per_unit
        realized = new_held * log_ret - cost

        ts = str(fv["timestamp"].iloc[end_idx]) if "timestamp" in fv.columns else str(end_idx)
        entries.append(JournalEntry(
            timestamp=ts,
            asset=asset,
            target_position=float(target),
            held_position=float(new_held),
            delta=float(new_held - held),
            realized_log_return=float(realized),
            selection_sharpe=float(sel_sharpe),
            quality_pass=q_pass,
        ))
        held = new_held

    # Summary
    rets = np.array([e.realized_log_return for e in entries], dtype=float)
    cum = float(np.exp(rets.sum()) - 1.0)
    sd = rets.std(ddof=0)
    sharpe = float(rets.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    equity = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0).min() if len(equity) else 0.0
    n_trades = int(sum(1 for e in entries if abs(e.delta) >= min_trade_size))
    avg_exp = float(np.mean([abs(e.held_position) for e in entries])) if entries else 0.0

    summary = JournalSummary(
        asset_count=1, n_steps=len(entries),
        aggregate_log_return=float(rets.sum()),
        aggregate_net_return=cum,
        aggregate_sharpe=sharpe,
        aggregate_max_drawdown=float(dd),
        n_trades_per_asset={asset: n_trades},
        avg_exposure=avg_exp,
    )

    if output_dir is not None:
        _write_journal(asset, entries, summary, Path(output_dir))

    return entries, summary


def _make_temp_csv(df: pd.DataFrame, original_path: Path) -> str:
    """Write a slice to a temp CSV next to the original."""
    tmp = original_path.parent / f".paper_tmp_{original_path.stem}.csv"
    df.to_csv(tmp, index=False)
    return str(tmp)


def _write_journal(asset: str, entries: list[JournalEntry],
                   summary: JournalSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(e) for e in entries]).to_csv(
        out_dir / f"journal_{asset}.csv", index=False,
    )
    (out_dir / f"journal_{asset}.json").write_text(
        json.dumps(asdict(summary), indent=2, default=float),
        encoding="utf-8",
    )


def run_journal_portfolio(
    configs: list[dict] | None = None,
    start_bar_offset: int = -300,
    target_annual_vol: float = 0.30,
    output_dir: str | Path = "reports/paper_journal",
    **kwargs,
) -> JournalSummary:
    """Run paper-trading journal across all portfolio assets and aggregate."""
    cfgs = configs or LIVE_ASSET_CONFIGS
    all_entries: dict[str, list[JournalEntry]] = {}

    for c in cfgs:
        asset = Path(c["csv"]).stem
        print(f"-> {asset} journaling...", flush=True)
        try:
            entries, summary = simulate_paper_trading(
                c["csv"], include_funding=c["include_funding"],
                start_bar_offset=start_bar_offset,
                output_dir=output_dir,
                **kwargs,
            )
        except Exception as exc:
            print(f"   skip: {exc}")
            continue
        all_entries[asset] = entries
        print(f"   steps={summary.n_steps}  "
              f"trades={summary.n_trades_per_asset[asset]}  "
              f"netRet={summary.aggregate_net_return*100:+.1f}%  "
              f"Sharpe={summary.aggregate_sharpe:+.2f}")

    # Build a portfolio time-series by aligning timestamps
    if not all_entries:
        raise RuntimeError("No journals could be generated.")

    series = {}
    for asset, entries in all_entries.items():
        s = pd.Series(
            [e.realized_log_return for e in entries],
            index=pd.to_datetime([e.timestamp for e in entries]),
            name=asset,
        )
        s = s[~s.index.duplicated(keep="first")]
        series[asset] = s

    df = pd.concat(series, axis=1).sort_index()
    # Vol-scale per asset against the journal's own returns
    scalers = {}
    for a in df.columns:
        v = df[a].dropna()
        if len(v) > 20:
            head_vol = v.iloc[:min(len(v) // 2, 100)].std(ddof=0) * np.sqrt(
                kwargs.get("bars_per_year", 2190)
            )
            scalers[a] = float(target_annual_vol / head_vol) if head_vol > 0 else 1.0
        else:
            scalers[a] = 1.0
    scaled = df.copy()
    for a, k in scalers.items():
        scaled[a] *= k

    n_active = scaled.notna().sum(axis=1).replace(0, np.nan)
    weighted = scaled.fillna(0.0).div(n_active.fillna(1), axis=0)
    port_ret = weighted.sum(axis=1)

    equity = np.exp(port_ret.cumsum())
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    sd = port_ret.std(ddof=0)
    sharpe = float(port_ret.mean() / sd * np.sqrt(
        kwargs.get("bars_per_year", 2190)
    )) if sd > 0 else 0.0

    out_dir = Path(output_dir)
    pd.DataFrame({
        "portfolio_log_return": port_ret,
        "equity": equity,
        "drawdown": equity / peak - 1.0,
        "n_active": n_active,
    }).to_csv(out_dir / "portfolio_journal.csv")

    n_trades_total = {a: sum(1 for e in all_entries[a]
                              if abs(e.delta) >= kwargs.get("min_trade_size", 0.05))
                      for a in all_entries}

    portfolio_summary = JournalSummary(
        asset_count=len(all_entries),
        n_steps=len(port_ret),
        aggregate_log_return=float(port_ret.sum()),
        aggregate_net_return=float(np.exp(port_ret.sum()) - 1.0),
        aggregate_sharpe=sharpe,
        aggregate_max_drawdown=float(dd),
        n_trades_per_asset=n_trades_total,
        avg_exposure=float(n_active.mean() / len(all_entries)),
    )
    (out_dir / "portfolio_journal.json").write_text(
        json.dumps({
            **asdict(portfolio_summary),
            "vol_scalers": scalers,
        }, indent=2, default=float),
        encoding="utf-8",
    )

    return portfolio_summary
