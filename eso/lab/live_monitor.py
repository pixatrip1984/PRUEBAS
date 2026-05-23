"""Live monitor + circuit breaker.

Tracks realized P&L of the live portfolio against backtest expectations.
Emits one of three signals per check:

  GO    — running within the 95% CI band of the backtest. Continue.
  WARN  — outside the typical band but inside the stop thresholds.
          Continue with caution; investigate.
  STOP  — one of the explicit stop triggers fired. Halt trading.

The monitor maintains a persistent JSON log of every observation:
    timestamp, target_positions, actual_positions, marks, realized_pnl

Two CLI modes:
    seed   — bootstrap the log from the paper-trading journal (so the
             very first check has a baseline to compare against).
    update — append a new observation and emit the GO/WARN/STOP signal.
    status — print the current state without appending.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# Backtest expectation from the latest portfolio_voltarget_baseline run:
#   Sharpe point     +3.14
#   Sharpe 95% CI    [+0.98, +5.49]
#   Max DD           -11.4%
# These define the GO/WARN/STOP bands.
BACKTEST_EXPECTATION = {
    "sharpe_point": 3.14,
    "sharpe_lo": 0.98,
    "sharpe_hi": 5.49,
    "max_dd_backtest": -0.114,
    # Stop triggers (more permissive than the CI to avoid false STOPs on
    # short windows; STOP requires a sustained breach).
    "stop_dd_threshold": -0.20,   # 1.75x backtest max DD
    "stop_sharpe_threshold": 0.30,  # well below the CI lower bound
    "warn_sharpe_threshold": 0.80,  # near the CI lower bound
    "min_bars_for_stop": 60,        # at least 60 bars (~10 days at 4h) before STOP can fire
}


@dataclass
class LiveObservation:
    timestamp: str
    target_positions: dict       # asset -> target exposure (live_signal output)
    actual_positions: dict       # asset -> actual exposure (post-quality-gate)
    marks: dict                  # asset -> close price at this bar
    realized_log_returns: dict   # asset -> log return on this bar (after costs)
    portfolio_log_return: float  # weighted sum of asset log returns
    notes: str = ""


@dataclass
class MonitorStatus:
    n_observations: int
    last_timestamp: str
    realized_total_return: float
    realized_sharpe: float
    realized_max_drawdown: float
    bars_in_drawdown: int
    bars_below_warn_sharpe: int
    signal: Literal["GO", "WARN", "STOP"]
    reasons: list = field(default_factory=list)
    expectation: dict = field(default_factory=lambda: dict(BACKTEST_EXPECTATION))


def _empty_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"observations": []}, indent=2),
                    encoding="utf-8")


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("observations", [])


def _write_log(path: Path, observations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"observations": observations}, indent=2, default=float),
        encoding="utf-8",
    )


def seed_from_paper_journal(
    journal_csv: str | Path,
    log_path: str | Path,
) -> int:
    """Bootstrap the monitor's log from a paper-trading journal CSV.

    The journal CSV (e.g. reports/paper_journal_strict/portfolio_journal.csv)
    has columns: portfolio_log_return, equity, drawdown, n_active.
    Each row becomes a synthetic observation in the log.
    """
    df = pd.read_csv(journal_csv, index_col=0)
    df.index = pd.to_datetime(df.index)
    observations = []
    for ts, row in df.iterrows():
        observations.append(asdict(LiveObservation(
            timestamp=ts.isoformat(),
            target_positions={},
            actual_positions={},
            marks={},
            realized_log_returns={},
            portfolio_log_return=float(row["portfolio_log_return"]),
            notes="seeded from paper journal",
        )))
    _write_log(Path(log_path), observations)
    return len(observations)


def append_observation(
    log_path: str | Path,
    obs: LiveObservation,
) -> int:
    """Append a new live observation to the log. Returns total count."""
    path = Path(log_path)
    observations = _read_log(path)
    observations.append(asdict(obs))
    _write_log(path, observations)
    return len(observations)


def compute_status(
    log_path: str | Path,
    bars_per_year: int = 2190,
    expectation: dict | None = None,
) -> MonitorStatus:
    """Aggregate the log into the current GO/WARN/STOP status."""
    exp = expectation or BACKTEST_EXPECTATION
    observations = _read_log(Path(log_path))
    if not observations:
        return MonitorStatus(
            n_observations=0, last_timestamp="",
            realized_total_return=0.0, realized_sharpe=0.0,
            realized_max_drawdown=0.0, bars_in_drawdown=0,
            bars_below_warn_sharpe=0,
            signal="GO",
            reasons=["no observations yet — first run"],
        )

    rets = np.array([o["portfolio_log_return"] for o in observations],
                    dtype=float)
    equity = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(equity)
    dd_series = equity / peak - 1.0
    bars_in_dd = int((dd_series < -0.01).sum())
    max_dd = float(dd_series.min())

    sd = float(rets.std(ddof=0))
    sharpe = float(rets.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0

    # Rolling Sharpe over last 60 bars to detect collapse
    bars_below_warn = 0
    if len(rets) >= 30:
        recent = rets[-60:] if len(rets) >= 60 else rets[-30:]
        recent_sd = recent.std(ddof=0)
        if recent_sd > 0:
            recent_sharpe = recent.mean() / recent_sd * np.sqrt(bars_per_year)
            if recent_sharpe < exp["warn_sharpe_threshold"]:
                bars_below_warn = len(recent)

    reasons = []
    signal: Literal["GO", "WARN", "STOP"] = "GO"

    # STOP triggers (require min_bars to avoid early-noise false positives)
    if len(rets) >= exp["min_bars_for_stop"]:
        if max_dd <= exp["stop_dd_threshold"]:
            signal = "STOP"
            reasons.append(
                f"Drawdown {max_dd*100:+.1f}% breaches stop "
                f"threshold {exp['stop_dd_threshold']*100:+.1f}%"
            )
        if sharpe < exp["stop_sharpe_threshold"]:
            signal = "STOP"
            reasons.append(
                f"Realized Sharpe {sharpe:+.2f} below stop "
                f"threshold {exp['stop_sharpe_threshold']:+.2f}"
            )

    if signal != "STOP":
        # WARN triggers
        if sharpe < exp["warn_sharpe_threshold"]:
            signal = "WARN"
            reasons.append(
                f"Realized Sharpe {sharpe:+.2f} below CI-lower "
                f"warn threshold {exp['warn_sharpe_threshold']:+.2f}"
            )
        elif max_dd <= -0.10:
            signal = "WARN"
            reasons.append(
                f"Drawdown {max_dd*100:+.1f}% approaching backtest "
                f"max {exp['max_dd_backtest']*100:+.1f}%"
            )

    if signal == "GO":
        reasons.append(
            f"Sharpe {sharpe:+.2f} within CI band "
            f"[{exp['sharpe_lo']:+.2f}, {exp['sharpe_hi']:+.2f}]"
        )

    return MonitorStatus(
        n_observations=len(observations),
        last_timestamp=observations[-1]["timestamp"],
        realized_total_return=float(np.exp(rets.sum()) - 1.0),
        realized_sharpe=sharpe,
        realized_max_drawdown=max_dd,
        bars_in_drawdown=bars_in_dd,
        bars_below_warn_sharpe=bars_below_warn,
        signal=signal,
        reasons=reasons,
        expectation=exp,
    )


def print_status(status: MonitorStatus) -> None:
    """Human-readable dashboard."""
    banner = {
        "GO":   "[ GO   ] strategy running within expectation",
        "WARN": "[ WARN ] divergence from backtest — investigate",
        "STOP": "[ STOP ] stop trigger fired — halt new trades",
    }[status.signal]
    print("\n" + "=" * 60)
    print(banner)
    print("=" * 60)
    print(f"Observations:      {status.n_observations}")
    print(f"Last timestamp:    {status.last_timestamp}")
    print(f"Realized return:   {status.realized_total_return*100:+.2f}%")
    print(f"Realized Sharpe:   {status.realized_sharpe:+.2f}  "
          f"(expectation CI [{status.expectation['sharpe_lo']:+.2f}, "
          f"{status.expectation['sharpe_hi']:+.2f}])")
    print(f"Max drawdown:      {status.realized_max_drawdown*100:+.2f}%  "
          f"(backtest max {status.expectation['max_dd_backtest']*100:+.1f}%, "
          f"STOP at {status.expectation['stop_dd_threshold']*100:+.1f}%)")
    print(f"Bars in drawdown:  {status.bars_in_drawdown}")
    print("\nReasons:")
    for r in status.reasons:
        print(f"  - {r}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Live monitor / circuit breaker")
    sub = p.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Bootstrap the log from a paper journal")
    p_seed.add_argument("--journal-csv", required=True)
    p_seed.add_argument("--log", default="reports/live_monitor/log.json")

    p_stat = sub.add_parser("status", help="Print current monitor status")
    p_stat.add_argument("--log", default="reports/live_monitor/log.json")

    p_obs = sub.add_parser("observe",
                           help="Append one live observation from a JSON file")
    p_obs.add_argument("observation_json",
                       help="Path to a JSON with the keys of LiveObservation")
    p_obs.add_argument("--log", default="reports/live_monitor/log.json")

    args = p.parse_args(argv)

    if args.command == "seed":
        n = seed_from_paper_journal(args.journal_csv, args.log)
        print(f"Seeded {n} observations into {args.log}")
        st = compute_status(args.log)
        print_status(st)
        return 0

    if args.command == "status":
        st = compute_status(args.log)
        print_status(st)
        return 0 if st.signal != "STOP" else 3  # exit code 3 = STOP

    if args.command == "observe":
        payload = json.loads(Path(args.observation_json).read_text(encoding="utf-8"))
        obs = LiveObservation(**payload)
        n = append_observation(args.log, obs)
        print(f"Logged observation {n}: {obs.timestamp}")
        st = compute_status(args.log)
        print_status(st)
        return 0 if st.signal != "STOP" else 3

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
