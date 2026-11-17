"""Causal microstructure fingerprints for OHLCV/order-flow data."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _numeric(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        values = pd.to_numeric(df[name], errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _returns(df: pd.DataFrame) -> pd.Series:
    close = _numeric(df, "close", 0.0)
    safe = close.where(close > 0)
    out = np.log(safe).diff()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _range_fraction(df: pd.DataFrame, returns: pd.Series) -> pd.Series:
    close = _numeric(df, "close", 0.0).abs()
    if {"high", "low"}.issubset(df.columns):
        high = _numeric(df, "high", 0.0)
        low = _numeric(df, "low", 0.0)
        rng = (high - low).abs() / close.replace(0.0, np.nan)
    else:
        rng = returns.abs()
    return rng.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    minp = max(2, min(window, window // 3 if window >= 3 else 2))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _signed_imbalance(df: pd.DataFrame, returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    volume = _numeric(df, "volume", 0.0).clip(lower=0.0)
    if {"taker_buy_volume", "taker_sell_volume"}.issubset(df.columns):
        buy = _numeric(df, "taker_buy_volume", 0.0).clip(lower=0.0)
        sell = _numeric(df, "taker_sell_volume", 0.0).clip(lower=0.0)
        gross = (buy + sell).where((buy + sell) > 0.0, volume)
        signed = buy - sell
    elif "delta" in df.columns:
        signed = _numeric(df, "delta", 0.0)
        gross = volume
    else:
        direction = np.sign(returns).replace(0.0, np.nan).ffill().fillna(0.0)
        signed = direction * volume
        gross = volume
    ratio = signed / gross.replace(0.0, np.nan)
    return (
        ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
        signed.replace([np.inf, -np.inf], np.nan).fillna(0.0),
    )


def _compression(range_fraction: pd.Series, window: int) -> pd.Series:
    baseline = range_fraction.rolling(window, min_periods=1).median()
    ratio = baseline / range_fraction.replace(0.0, np.nan)
    zero_now = range_fraction.eq(0.0)
    score = np.log1p(ratio.clip(lower=0.0, upper=20.0))
    score = score.where(~zero_now, np.log1p(20.0))
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_stealth_accumulation_index(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Score high/one-sided volume absorbed inside unusually compressed price ranges.

    The score is non-negative, finite, and strictly backward-looking. It is a
    structural fingerprint, not a claim about trader identity.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    returns = _returns(df)
    volume = _numeric(df, "volume", 0.0).clip(lower=0.0)
    rng = _range_fraction(df, returns)
    imbalance, _ = _signed_imbalance(df, returns)

    vol_z = _rolling_zscore(np.log1p(volume), window).clip(lower=0.0)
    volume_pressure = np.log1p(
        volume / volume.rolling(window, min_periods=1).median().replace(0.0, np.nan)
    )
    volume_pressure = (
        volume_pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    )
    absorption = 0.5 + 0.5 * imbalance.abs()
    score = (1.0 + vol_z) * volume_pressure * _compression(rng, window) * absorption
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    score.name = "stealth_accumulation_index"
    return score


def compute_flow_toxicity_vpin(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return VPIN-like rolling toxicity and a causal Kyle-lambda impact proxy.

    VPIN is absolute rolling signed-volume imbalance divided by rolling gross
    volume and therefore remains in [0, 1]. kyle_lambda is a rolling median of
    absolute log-return per unit notional (or base volume when price is absent).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    returns = _returns(df)
    volume = _numeric(df, "volume", 0.0).clip(lower=0.0)
    _, signed = _signed_imbalance(df, returns)

    gross_roll = volume.rolling(window, min_periods=1).sum()
    signed_roll = signed.rolling(window, min_periods=1).sum().abs()
    vpin = (
        signed_roll / gross_roll.replace(0.0, np.nan)
    ).clip(0.0, 1.0).fillna(0.0)

    close = _numeric(df, "close", 0.0).abs()
    notional = (close * volume).where(close.gt(0.0), volume)
    instant_lambda = returns.abs() / notional.replace(0.0, np.nan)
    kyle = instant_lambda.rolling(window, min_periods=1).median()
    kyle = (
        kyle.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    )

    return pd.DataFrame({"vpin": vpin, "kyle_lambda": kyle}, index=df.index)


def compute_order_splitting_fingerprint(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Detect persistent directional flow fragmented over many executions/bars."""
    if window < 2:
        raise ValueError("window must be >= 2")
    returns = _returns(df)
    volume = _numeric(df, "volume", 0.0).clip(lower=0.0)
    rng = _range_fraction(df, returns)
    imbalance, _ = _signed_imbalance(df, returns)

    persistence = imbalance.rolling(window, min_periods=1).mean().abs()
    same_sign = np.sign(imbalance).replace(0.0, np.nan)
    directional_consistency = same_sign.rolling(window, min_periods=1).apply(
        lambda x: abs(np.nansum(x)) / max(np.sum(np.isfinite(x)), 1), raw=True
    ).fillna(0.0)

    if "n_trades" in df.columns:
        trades = _numeric(df, "n_trades", 0.0).clip(lower=0.0)
        ticket = volume / trades.replace(0.0, np.nan)
        ticket_z = _rolling_zscore(np.log1p(ticket.fillna(0.0)), window)
        trade_burst = _rolling_zscore(np.log1p(trades), window).clip(lower=0.0)
        fragmentation = (trade_burst + (-ticket_z).clip(lower=0.0)) / 2.0
    else:
        fragmentation = (
            _rolling_zscore(np.log1p(volume), window).clip(lower=0.0) / 2.0
        )

    vol_pressure = _rolling_zscore(np.log1p(volume), window).clip(lower=0.0)
    score = (
        (0.5 * persistence + 0.5 * directional_consistency)
        * (1.0 + fragmentation + 0.5 * vol_pressure)
        * _compression(rng, window)
    )
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    score.name = "order_splitting_fingerprint"
    return score


def compute_microstructure_fingerprints(
    df: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """Compute all microstructure fingerprints with index preserved."""
    toxicity = compute_flow_toxicity_vpin(df, window=window)
    out = pd.DataFrame(index=df.index)
    out["stealth_accumulation_index"] = compute_stealth_accumulation_index(df, window)
    out["vpin"] = toxicity["vpin"]
    out["kyle_lambda"] = toxicity["kyle_lambda"]
    out["order_splitting_fingerprint"] = compute_order_splitting_fingerprint(df, window)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
