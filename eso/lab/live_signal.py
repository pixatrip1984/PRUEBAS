"""Live signal generator — converts the backtested strategy into a tradable function.

For each asset:
  1. Load the CSV (must include enough history for UMAP training + walk-forward
     selection window).
  2. Build the causal feature vector (CausalCyclePhase uses first 50% to fit).
  3. Run a "selection sweep" on the trailing N bars to pick (gate, signal, enter).
  4. Apply gated_phase_threshold_strategy to the most recent bars.
  5. Output the position at the FINAL bar.

The output is a dict[asset, signal] where signal includes the position,
the metadata, and a confidence proxy (selection window Sharpe).

Usage:
    from eso.lab.live_signal import get_live_positions
    positions = get_live_positions(LIVE_ASSET_CONFIGS)

Or via CLI:
    python -m eso.lab.live_signal --output current_positions.json

This is meant to be re-run periodically (e.g. every 4h on the bar close)
with the latest data appended to each CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eso.backtest import BacktestConfig, gated_phase_threshold_strategy, run_backtest
from eso.lab.multi_asset_sweep import build_alt_feature_vector
from eso.lab.walk_forward import SWEEP_GRID


# Production asset configuration — derived from the walk-forward sessions
LIVE_ASSET_CONFIGS = [
    {"csv": "data/VVVUSDT_4h.csv",    "include_funding": False},
    {"csv": "data/STGUSDT_4h.csv",    "include_funding": False},
    {"csv": "data/MORPHOUSDT_4h.csv", "include_funding": False},
    {"csv": "data/GNOUSDT_4h.csv",    "include_funding": True},
    {"csv": "data/GRASSUSDT_4h.csv",  "include_funding": False},
]


@dataclass
class LiveSignal:
    asset: str
    timestamp: str
    current_position: float        # in [-1, 1], dimensionless target exposure
    last_close: float
    selected_gate: float
    selected_signal: str
    selected_enter: float
    selection_sharpe: float        # confidence proxy
    selection_window_bars: int
    n_recent_active_bars: int       # how many recent bars had a non-zero position


@dataclass
class LivePortfolioSignal:
    generated_at: str
    target_annual_vol: float
    per_asset: list                 # list of LiveSignal as dicts
    vol_scalers: dict
    portfolio_positions: dict       # asset -> final scaled position
    notes: list = field(default_factory=list)


def _select_best_params(
    fv: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[float, str, float, float]:
    """Pick (gate, signal_col, enter) with highest net Sharpe on this slice."""
    best = (None, float("-inf"))
    prices = fv["close"]
    for gp, sc, et in SWEEP_GRID:
        if sc not in fv.columns:
            continue
        pos = gated_phase_threshold_strategy(
            fv, signal_col=sc, gate_col="ring_radius",
            gate_percentile=gp, gate_window=min(300, len(fv) // 3),
            enter_threshold=et, exit_threshold=et / 3,
            allow_short=True,
        )
        bt = run_backtest(prices, pos, cfg)
        s = bt.metrics["net_sharpe"]
        if not np.isnan(s) and s > best[1]:
            best = ((gp, sc, et), s)
    if best[0] is None:
        return 0.50, "cos_theta_24h", 0.30, float("nan")
    gp, sc, et = best[0]
    return gp, sc, et, best[1]


def asset_live_signal(
    csv_path: str | Path,
    include_funding: bool = False,
    selection_window_bars: int = 700,
    bars_per_year: int = 2190,
    fee_bps: float = 5.5,
    slippage_bps: float = 2.0,
    seed: int = 42,
) -> LiveSignal | None:
    """Compute the current position for one asset.

    selection_window_bars is the trailing slice used to pick parameters.
    Default 700 bars ≈ 4 months at 4h, similar to one walk-forward fold.
    """
    csv_path = Path(csv_path)
    asset = csv_path.stem
    df = pd.read_csv(csv_path)
    if len(df) < 400:
        return None

    fv = build_alt_feature_vector(df, seed=seed, include_funding=include_funding)
    if len(fv) < selection_window_bars + 30:
        # Fall back to whatever's available, but flag the warning
        selection_window_bars = max(200, len(fv) // 2)

    # Pick params on the trailing selection window (excluding the very last
    # `recent_eval_bars` which are what we're going to "live trade").
    recent_eval_bars = min(50, len(fv) // 10)  # last ~50 bars to apply the chosen rule
    sel_end = len(fv) - recent_eval_bars
    sel_start = max(0, sel_end - selection_window_bars)
    fv_sel = fv.iloc[sel_start:sel_end].copy()

    cfg = BacktestConfig(
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        bars_per_year=bars_per_year, allow_short=True,
    )
    gp, sc, et, sel_sharpe = _select_best_params(fv_sel, cfg)

    # Apply chosen rule to the most recent slice (selection_window + eval)
    fv_apply = fv.iloc[sel_start:].copy()
    pos_series = gated_phase_threshold_strategy(
        fv_apply, signal_col=sc, gate_col="ring_radius",
        gate_percentile=gp, gate_window=min(300, len(fv_apply) // 3),
        enter_threshold=et, exit_threshold=et / 3,
        allow_short=True,
    )

    current = float(pos_series.iloc[-1])
    last_close = float(fv["close"].iloc[-1])
    ts = str(fv["timestamp"].iloc[-1]) if "timestamp" in fv.columns else ""
    recent_active = int((pos_series.iloc[-recent_eval_bars:] != 0).sum())

    return LiveSignal(
        asset=asset,
        timestamp=ts,
        current_position=current,
        last_close=last_close,
        selected_gate=gp,
        selected_signal=sc,
        selected_enter=et,
        selection_sharpe=float(sel_sharpe),
        selection_window_bars=len(fv_sel),
        n_recent_active_bars=recent_active,
    )


def get_live_positions(
    configs: list[dict] | None = None,
    target_annual_vol: float = 0.30,
    output_path: str | Path | None = None,
    selection_window_bars: int = 700,
) -> LivePortfolioSignal:
    """Produce the current portfolio's per-asset target positions.

    Applies the same vol-targeting logic as the backtest: each asset is
    sized inversely to its realised vol. The trader then scales these
    target exposures by their account size.

    Output JSON is suitable for piping into an execution layer.
    """
    cfgs = configs or LIVE_ASSET_CONFIGS
    notes = []
    signals: list[LiveSignal] = []

    for c in cfgs:
        try:
            sig = asset_live_signal(
                c["csv"], include_funding=c["include_funding"],
                selection_window_bars=selection_window_bars,
            )
        except Exception as exc:
            notes.append(f"{Path(c['csv']).stem}: failed — {exc}")
            continue
        if sig is None:
            notes.append(f"{Path(c['csv']).stem}: insufficient data")
            continue
        signals.append(sig)

    if not signals:
        raise RuntimeError("No live signals could be generated.")

    # Estimate per-asset vol from the last ~200 bars of feature vector returns
    vol_scalers = {}
    for s in signals:
        try:
            df = pd.read_csv(Path(next(c["csv"] for c in cfgs if Path(c["csv"]).stem == s.asset)))
            recent = np.log(df["close"].astype(float)).diff().dropna().tail(200)
            vol = recent.std(ddof=0) * np.sqrt(s_bars_per_year(df) or 2190)
            scaler = float(target_annual_vol / vol) if vol > 0 else 1.0
            vol_scalers[s.asset] = scaler
        except Exception:
            vol_scalers[s.asset] = 1.0

    # Final positions: target * scaler, capped at +/- 1
    portfolio_positions = {}
    for s in signals:
        scaler = vol_scalers.get(s.asset, 1.0)
        # Equal-weight across N assets
        weight = 1.0 / len(signals)
        scaled = float(np.clip(s.current_position * scaler * weight, -1.0, 1.0))
        portfolio_positions[s.asset] = scaled

    report = LivePortfolioSignal(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        target_annual_vol=target_annual_vol,
        per_asset=[asdict(s) for s in signals],
        vol_scalers=vol_scalers,
        portfolio_positions=portfolio_positions,
        notes=notes,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2, default=float),
                       encoding="utf-8")

    return report


def s_bars_per_year(df: pd.DataFrame) -> int | None:
    """Infer the bar duration from timestamps and return bars/year, or None."""
    if "timestamp" not in df.columns or len(df) < 2:
        return None
    ts = pd.to_datetime(df["timestamp"].iloc[:3])
    diffs = ts.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return None
    median = float(diffs.median())
    if median <= 0:
        return None
    return int(365 * 24 * 3600 / median)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Generate live portfolio positions")
    p.add_argument("--output", default="reports/live_signal.json",
                   help="Where to write the JSON report")
    p.add_argument("--target-vol", type=float, default=0.30,
                   help="Target annual portfolio vol (default 0.30)")
    p.add_argument("--selection-window", type=int, default=700,
                   help="Trailing bars for parameter selection")
    args = p.parse_args(argv)

    report = get_live_positions(
        target_annual_vol=args.target_vol,
        output_path=args.output,
        selection_window_bars=args.selection_window,
    )

    print(f"Live signal generated at {report.generated_at}")
    print(f"Output: {args.output}\n")
    print("Per-asset current positions:")
    for s in report.per_asset:
        marker = "LONG " if s["current_position"] > 0.05 else ("SHORT" if s["current_position"] < -0.05 else "FLAT ")
        print(f"  {s['asset']:18s}  {marker}  pos={s['current_position']:+.2f}  "
              f"@{s['last_close']:.4f}  "
              f"(sel_Sharpe={s['selection_sharpe']:+.2f}, "
              f"signal={s['selected_signal']})")

    print("\nVol-scaled portfolio positions (equal-weight × 1/n × vol_scaler):")
    for asset, pos in report.portfolio_positions.items():
        print(f"  {asset:18s}  target_exposure = {pos:+.4f}")

    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
