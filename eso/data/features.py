"""Financial feature engineering for ESO — converts raw OHLCV + microstructure to stationary signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_financial_features(
    df: pd.DataFrame,
    vol_windows: tuple[int, ...] = (5, 20),
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Return a new DataFrame with stationary financial features derived from OHLCV + microstructure.

    Required columns: close
    Optional (enriched if present): open, high, low, volume, vwap,
                                    taker_buy_volume, taker_sell_volume, delta
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)

    # ── trend-removed price signal ──────────────────────────────────────────
    log_close = np.log(close.replace(0, np.nan))
    out["log_return"] = log_close.diff()
    out["log_return_2"] = log_close.diff(2)

    # ── intra-bar range (normalised volatility proxy) ───────────────────────
    if {"high", "low"}.issubset(df.columns):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        out["hl_range"] = (high - low) / close.replace(0, np.nan)

    if "open" in df.columns:
        out["body"] = (close - df["open"].astype(float)) / close.replace(0, np.nan)

    # ── rolling volatility ───────────────────────────────────────────────────
    lr = out["log_return"]
    for w in vol_windows:
        out[f"vol_{w}"] = lr.rolling(w, min_periods=w // 2).std()

    # ── rolling z-score of log_return ────────────────────────────────────────
    out["lr_z20"] = (lr - lr.rolling(20, min_periods=10).mean()) / (
        lr.rolling(20, min_periods=10).std().replace(0, np.nan)
    )

    # ── microstructure: VWAP deviation ──────────────────────────────────────
    if "vwap" in df.columns:
        vwap = df["vwap"].astype(float)
        out["vwap_dev"] = (close - vwap) / close.replace(0, np.nan)

    # ── microstructure: volume / taker imbalance ─────────────────────────────
    if {"taker_buy_volume", "taker_sell_volume", "volume"}.issubset(df.columns):
        vol = df["volume"].astype(float)
        buy = df["taker_buy_volume"].astype(float)
        sell = df["taker_sell_volume"].astype(float)
        out["volume_imbalance"] = (buy - sell) / (vol + eps)
        out["taker_ratio"] = buy / (vol + eps)

    if "delta" in df.columns and "volume" in df.columns:
        vol = df["volume"].astype(float)
        out["delta_norm"] = df["delta"].astype(float) / (vol + eps)

    # ── rolling volume ratio (activity proxy) ────────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20, min_periods=10).mean()
        out["vol_ratio"] = vol / (vol_ma + eps)

    if "n_trades" in df.columns:
        trades = df["n_trades"].astype(float)
        trades_ma = trades.rolling(20, min_periods=10).mean()
        out["trades_ratio"] = trades / (trades_ma + eps)

    # ── drop initial NaN rows (from diffs and rolling windows) ──────────────
    out = out.dropna(how="all").dropna(subset=["log_return"])

    return out


def prepare_btc_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Binance Klines DataFrames (1m/3m/5m/15m) to the column schema
    expected by build_financial_features().

    Maps: taker_buy_base_asset_volume -> taker_buy_volume,
          derives taker_sell_volume = volume - taker_buy,
          derives vwap = quote_asset_volume / volume,
          derives delta = taker_buy - taker_sell,
          renames number_of_trades -> n_trades.
    """
    out = df.copy()
    rename = {
        "taker_buy_base_asset_volume": "taker_buy_volume",
        "number_of_trades": "n_trades",
        "quote_asset_volume": "volume_usd",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if "taker_buy_volume" in out.columns and "volume" in out.columns:
        out["taker_sell_volume"] = out["volume"] - out["taker_buy_volume"]
        out["delta"] = out["taker_buy_volume"] - out["taker_sell_volume"]

    if "vwap" not in out.columns and "volume_usd" in out.columns and "volume" in out.columns:
        eps = 1e-9
        out["vwap"] = out["volume_usd"] / (out["volume"] + eps)

    return out


def feature_column_groups() -> dict[str, list[str]]:
    """Return named column groups for focused ESO experiments."""
    return {
        "returns_only": ["log_return", "log_return_2"],
        "returns_vol": ["log_return", "vol_5", "vol_20", "lr_z20"],
        "microstructure": ["vwap_dev", "volume_imbalance", "taker_ratio", "delta_norm"],
        "full": [
            "log_return", "log_return_2", "hl_range", "body",
            "vol_5", "vol_20", "lr_z20",
            "vwap_dev", "volume_imbalance", "taker_ratio", "delta_norm",
            "vol_ratio", "trades_ratio",
        ],
        "compact": ["log_return", "vol_20", "vwap_dev", "volume_imbalance"],
    }
