"""Export Bybit perpetual OHLCV + funding rates to CSV for ESO lab dissection.

Outputs:
  data/BTCUSDT_4h.csv   — BTC H4 reference (for alt relative features)
  data/<SYM>_4h.csv     — 10 alt symbols H4 + funding_rate column aligned to bar start

Usage:
  python scripts/export_bybit_data.py
  python scripts/export_bybit_data.py --days 500 --output-dir data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BYBIT_ROOT = ROOT.parent / "bybit-limit-order"
sys.path.insert(0, str(BYBIT_ROOT))
sys.path.insert(0, str(BYBIT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(str(BYBIT_ROOT / ".env"), override=True)

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

ALT_SYMBOLS = [
    "DRIFTUSDT", "GRASSUSDT", "SIGNUSDT", "NILUSDT", "VVVUSDT",
    "STGUSDT",   "GNOUSDT",   "ENSUSDT",  "POPCATUSDT", "MORPHOUSDT",
]

INTERVAL_4H = "240"  # 240 minutes = H4
MS_PER_4H = 4 * 60 * 60 * 1000


def _make_session() -> HTTP:
    key    = os.getenv("API_KEY") or os.getenv("BYBIT_API_KEY", "")
    secret = os.getenv("API_SECRET") or os.getenv("BYBIT_API_SECRET", "")
    return HTTP(testnet=False, api_key=key, api_secret=secret, recv_window=10000)


def _fetch_klines(bybit: HTTP, symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Paginate backwards and return a sorted OHLCV DataFrame."""
    now_ms = int(time.time() * 1000)
    end_ms = now_ms
    target_bars = days * (24 // (int(interval) // 60))  # bars per day
    rows = []
    max_pages = 30

    for page in range(max_pages):
        try:
            resp = bybit.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                end=end_ms,
                limit=1000,
            )
        except Exception as e:
            print(f"  [{symbol}] kline page {page} error: {e}")
            time.sleep(2)
            continue

        result = resp.get("result", {}).get("list", [])
        if not result:
            break

        for r in result:
            rows.append({
                "timestamp_ms": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })

        oldest_ms = min(int(r[0]) for r in result)
        end_ms = oldest_ms - 1

        if len(rows) >= target_bars:
            break

        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("timestamp_ms")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Keep only last `days` worth of data
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def _fetch_funding_rates(bybit: HTTP, symbol: str, days: int) -> pd.DataFrame:
    """Fetch all available funding rate history for the given day window."""
    now_ms = int(time.time() * 1000)
    end_ms = now_ms
    rows = []
    max_pages = 30

    for page in range(max_pages):
        try:
            resp = bybit.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                endTime=end_ms,
                limit=200,
            )
        except Exception as e:
            print(f"  [{symbol}] funding page {page} error: {e}")
            time.sleep(2)
            break

        result = resp.get("result", {}).get("list", [])
        if not result:
            break

        for r in result:
            rows.append({
                "timestamp_ms": int(r["fundingRateTimestamp"]),
                "funding_rate": float(r["fundingRate"]),
            })

        oldest_ms = min(int(r["fundingRateTimestamp"]) for r in result)
        cutoff_ms = now_ms - days * 86400 * 1000
        if oldest_ms <= cutoff_ms:
            break
        end_ms = oldest_ms - 1
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(rows).drop_duplicates("timestamp_ms")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    return df


def _merge_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Align funding rates to H4 bars using forward-fill (last known rate)."""
    if funding.empty:
        ohlcv["funding_rate"] = np.nan
        return ohlcv

    ohlcv = ohlcv.copy()
    funding = funding.sort_values("timestamp")

    # merge_asof: for each OHLCV bar, find the most recent funding rate <= bar timestamp
    merged = pd.merge_asof(
        ohlcv.sort_values("timestamp"),
        funding.rename(columns={"timestamp": "timestamp"}),
        on="timestamp",
        direction="backward",
    )
    return merged.sort_values("timestamp").reset_index(drop=True)


def export_symbol(bybit: HTTP, symbol: str, days: int, out_dir: Path) -> Path:
    """Fetch and write one symbol's H4 CSV. Returns output path."""
    print(f"  [{symbol}] fetching klines...")
    ohlcv = _fetch_klines(bybit, symbol, INTERVAL_4H, days)
    if ohlcv.empty:
        print(f"  [{symbol}] no kline data - skipping")
        return None

    print(f"  [{symbol}] fetching funding rates...")
    funding = _fetch_funding_rates(bybit, symbol, days)

    df = _merge_funding(ohlcv, funding)
    out_path = out_dir / f"{symbol}_4h.csv"
    df.to_csv(out_path, index=False)
    fr_count = df["funding_rate"].notna().sum()
    print(f"  [{symbol}] {len(df)} bars, {fr_count} with funding_rate -> {out_path.name}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Bybit data for ESO lab")
    parser.add_argument("--days", type=int, default=500, help="Days of history to fetch")
    parser.add_argument("--output-dir", default="data", help="Output directory (relative to repo root)")
    parser.add_argument("--alts-only", action="store_true", help="Skip BTC (already exists)")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bybit = _make_session()

    if not args.alts_only:
        print("\n[BTC H4] fetching...")
        btc_h4 = _fetch_klines(bybit, "BTCUSDT", INTERVAL_4H, args.days)
        if not btc_h4.empty:
            out = out_dir / "BTCUSDT_4h.csv"
            btc_h4.to_csv(out, index=False)
            print(f"  BTCUSDT H4: {len(btc_h4)} bars -> {out.name}")

    print(f"\n[Alts] fetching {len(ALT_SYMBOLS)} symbols x H4...")
    for sym in ALT_SYMBOLS:
        export_symbol(bybit, sym, args.days, out_dir)
        time.sleep(0.3)

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
