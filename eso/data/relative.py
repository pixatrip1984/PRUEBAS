"""Cross-asset relative feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_time_frame(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column not found: {timestamp_col}")
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    return out


def build_relative_market_features(
    asset: pd.DataFrame,
    reference: pd.DataFrame,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    reference_close_col: str = "close",
    windows: tuple[int, ...] = (24, 72),
    tolerance: str | pd.Timedelta | None = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Build causal asset-vs-reference features aligned by timestamp.

    The reference row is joined at the same timestamp, or the nearest previous
    reference timestamp when exact timestamps are missing. This supports common
    alt/BTC workflows where BTC is the macro reference.
    """
    if close_col not in asset.columns:
        raise ValueError(f"asset close column not found: {close_col}")
    if reference_close_col not in reference.columns:
        raise ValueError(f"reference close column not found: {reference_close_col}")

    left = _prepare_time_frame(asset, timestamp_col)
    right = _prepare_time_frame(reference, timestamp_col)
    right = right[[timestamp_col, reference_close_col]].rename(
        columns={reference_close_col: "reference_close"}
    )

    tol = pd.Timedelta(tolerance) if tolerance is not None else None
    merged = pd.merge_asof(
        left,
        right,
        on=timestamp_col,
        direction="backward",
        tolerance=tol,
    )
    merged = merged.dropna(subset=[close_col, "reference_close"]).copy()

    asset_close = pd.to_numeric(merged[close_col], errors="coerce").astype(float)
    ref_close = pd.to_numeric(merged["reference_close"], errors="coerce").astype(float)
    asset_lr = np.log(asset_close.replace(0, np.nan)).diff()
    ref_lr = np.log(ref_close.replace(0, np.nan)).diff()

    out = pd.DataFrame(index=merged.index)
    out[timestamp_col] = merged[timestamp_col].values
    out["close"] = asset_close.values
    out["reference_close"] = ref_close.values
    out["asset_log_return"] = asset_lr.values
    out["reference_log_return"] = ref_lr.values
    out["relative_log_return"] = (asset_lr - ref_lr).values
    out["price_ratio"] = (asset_close / (ref_close + eps)).values

    ratio_log = np.log(out["price_ratio"].replace(0, np.nan))
    for w in windows:
        min_periods = max(5, w // 2)
        out[f"relative_strength_{w}"] = ratio_log.diff(w).values
        corr = asset_lr.rolling(w, min_periods=min_periods).corr(ref_lr)
        ref_var = ref_lr.rolling(w, min_periods=min_periods).var()
        cov = asset_lr.rolling(w, min_periods=min_periods).cov(ref_lr)
        out[f"rolling_corr_ref_{w}"] = corr.values
        out[f"rolling_beta_ref_{w}"] = (cov / (ref_var + eps)).values
        ratio_mean = ratio_log.rolling(w, min_periods=min_periods).mean()
        ratio_std = ratio_log.rolling(w, min_periods=min_periods).std().replace(0, np.nan)
        out[f"price_ratio_z_{w}"] = ((ratio_log - ratio_mean) / ratio_std).values

    return out.dropna(subset=["asset_log_return", "reference_log_return"])
