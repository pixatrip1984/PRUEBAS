"""Deployment helpers: position diff, Bybit klines fetcher, signal quality gate.

Position diff:
    Given a live_signal.json (target positions) and a held_positions.json
    (current positions in your account), compute the delta to trade.
    Respects min_trade_size to suppress micro-rebalancing.

Bybit fetcher:
    Public klines endpoint (no API key needed for read-only data).
    Saves to the same CSV format expected by build_alt_feature_vector.

Signal quality gate:
    Per-asset filter that drops signals with low selection_sharpe — your
    own "veto" layer against the walker's worst picks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Bybit V5 public klines endpoint (no auth needed)
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
# Bybit V5 funding history (per perp pair)
BYBIT_FUNDING_URL = "https://api.bybit.com/v5/market/funding/history"


@dataclass
class TradeAction:
    asset: str
    held: float
    target: float
    delta: float
    action: str           # "BUY", "SELL", "HOLD"
    quality_pass: bool    # selection_sharpe above threshold
    notes: str = ""


def compute_trade_diff(
    live_signal_path: str | Path,
    held_positions: dict | None = None,
    min_trade_size: float = 0.05,
    min_selection_sharpe: float = 0.5,
) -> list[TradeAction]:
    """Compute the trade actions to bring held positions to target.

    Args:
        live_signal_path:        Path to JSON from get_live_positions().
        held_positions:          Dict {asset: current_exposure}. Missing = 0.
        min_trade_size:          |delta| below this returns HOLD action.
        min_selection_sharpe:    Assets with sel_Sharpe below this are
                                 flagged as quality_pass=False (don't trade).
    """
    held = held_positions or {}
    payload = json.loads(Path(live_signal_path).read_text(encoding="utf-8"))
    targets = payload["portfolio_positions"]
    per_asset = {s["asset"]: s for s in payload["per_asset"]}

    actions = []
    for asset, target in targets.items():
        h = float(held.get(asset, 0.0))
        delta = target - h
        sel_sharpe = per_asset.get(asset, {}).get("selection_sharpe", float("nan"))
        q_pass = (
            not np.isnan(sel_sharpe) and sel_sharpe >= min_selection_sharpe
        )

        if abs(delta) < min_trade_size:
            action = "HOLD"
            notes = "delta below min_trade_size"
        elif not q_pass:
            action = "HOLD"
            notes = (
                f"selection_sharpe {sel_sharpe:+.2f} below "
                f"min {min_selection_sharpe:+.2f}"
            )
        elif delta > 0:
            action = "BUY"
            notes = ""
        else:
            action = "SELL"
            notes = ""

        actions.append(TradeAction(
            asset=asset,
            held=h,
            target=target,
            delta=delta,
            action=action,
            quality_pass=q_pass,
            notes=notes,
        ))
    return actions


def write_trade_actions(actions: list[TradeAction], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actions": [asdict(a) for a in actions],
    }
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def fetch_bybit_klines(
    symbol: str,
    interval: str = "240",   # 240 minutes = 4h
    limit: int = 1000,
    category: str = "linear",
) -> pd.DataFrame:
    """Pull recent klines from Bybit public endpoint.

    Bybit returns up to 1000 bars per request. For longer histories you must
    paginate. This helper is intended for the periodic refresh use-case
    (append the latest N bars to your existing CSV).

    Args:
        symbol:    e.g. "STGUSDT"
        interval:  "1", "3", "5", "15", "30", "60", "120", "240", "360",
                   "720", "D", "M", "W" (numeric is minutes)
        limit:     1..1000
        category:  "linear" (USDT perps), "spot", "inverse"

    Returns:
        DataFrame with columns matching the local CSV schema:
        timestamp, open, high, low, close, volume
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Install 'requests' to fetch Bybit data: pip install requests"
        ) from exc

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if body.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {body.get('retMsg')}")

    raw = body["result"]["list"]
    # Bybit returns newest-first; reverse so it's ascending by time
    rows = list(reversed(raw))
    df = pd.DataFrame(rows, columns=[
        "start", "open", "high", "low", "close", "volume", "turnover",
    ])
    df["timestamp"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def refresh_asset_csv(
    symbol: str,
    csv_path: str | Path,
    interval: str = "240",
    fetch_limit: int = 200,
) -> int:
    """Append the latest bars from Bybit to an existing CSV.

    Returns the number of NEW rows appended (0 if already up-to-date).
    Preserves any extra columns (e.g. funding_rate) that the existing CSV
    has — those need a separate refresh path.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        # Cold start: just write the fetched data
        df = fetch_bybit_klines(symbol, interval=interval, limit=fetch_limit)
        df.to_csv(csv_path, index=False)
        return len(df)

    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    last_ts = existing["timestamp"].max()

    fresh = fetch_bybit_klines(symbol, interval=interval, limit=fetch_limit)
    new_rows = fresh[fresh["timestamp"] > last_ts]
    if new_rows.empty:
        return 0

    # Preserve any extra columns the existing CSV has (e.g. funding_rate).
    # Missing values in the new rows go in as NaN — the caller is responsible
    # for filling them if needed.
    extra_cols = [c for c in existing.columns if c not in new_rows.columns]
    for c in extra_cols:
        new_rows[c] = np.nan

    combined = pd.concat([existing, new_rows[existing.columns]], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined.to_csv(csv_path, index=False)
    return len(new_rows)


def print_trade_actions(actions: list[TradeAction]) -> None:
    """Human-readable summary."""
    print(f"{'ASSET':<18s} {'HELD':>8s} {'TARGET':>8s} {'DELTA':>8s}  ACTION  NOTES")
    for a in actions:
        marker = "OK " if a.quality_pass else "BAD"
        print(f"{a.asset:<18s} {a.held:>+8.3f} {a.target:>+8.3f} "
              f"{a.delta:>+8.3f}  {a.action:<6s} {marker} {a.notes}")
    total_target = sum(a.target for a in actions)
    print(f"\nTotal target exposure: {total_target:+.3f}  "
          f"({len([a for a in actions if a.action != 'HOLD'])} active trades)")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Deployment helpers")
    sub = p.add_subparsers(dest="command", required=True)

    p_diff = sub.add_parser("diff", help="Compute trade diff vs held positions")
    p_diff.add_argument("live_signal", help="Path to live_signal.json")
    p_diff.add_argument("--held", help="Path to held_positions.json (optional)")
    p_diff.add_argument("--min-trade-size", type=float, default=0.05)
    p_diff.add_argument("--min-sharpe", type=float, default=0.5)
    p_diff.add_argument("--output", help="Where to write trade actions JSON")

    p_fetch = sub.add_parser("fetch", help="Append latest klines from Bybit to a CSV")
    p_fetch.add_argument("symbol", help="e.g. STGUSDT")
    p_fetch.add_argument("csv_path")
    p_fetch.add_argument("--interval", default="240")
    p_fetch.add_argument("--limit", type=int, default=200)

    args = p.parse_args(argv)

    if args.command == "diff":
        held = {}
        if args.held:
            held = json.loads(Path(args.held).read_text(encoding="utf-8"))
        actions = compute_trade_diff(
            args.live_signal, held_positions=held,
            min_trade_size=args.min_trade_size,
            min_selection_sharpe=args.min_sharpe,
        )
        print_trade_actions(actions)
        if args.output:
            write_trade_actions(actions, args.output)
        return 0

    if args.command == "fetch":
        n = refresh_asset_csv(
            args.symbol, args.csv_path,
            interval=args.interval, fetch_limit=args.limit,
        )
        print(f"Appended {n} new rows to {args.csv_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
