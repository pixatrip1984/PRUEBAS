"""Context feature helpers for lab instruments."""

from __future__ import annotations

import pandas as pd

from eso.data.relative import build_relative_market_features


def append_reference_context(
    feature_frame: pd.DataFrame,
    raw_asset: pd.DataFrame,
    reference: pd.DataFrame | None,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    windows: tuple[int, ...] = (24, 72),
    tolerance: str | None = None,
) -> pd.DataFrame:
    """Append asset-vs-reference features to an existing feature frame."""
    if reference is None:
        return feature_frame
    if timestamp_col not in feature_frame.columns:
        raise ValueError("feature frame must include timestamp to append reference context")

    rel = build_relative_market_features(
        raw_asset,
        reference,
        timestamp_col=timestamp_col,
        close_col=close_col,
        reference_close_col=close_col,
        windows=windows,
        tolerance=tolerance,
    )
    keep_cols = [
        c for c in rel.columns
        if c not in {timestamp_col, "close"}
    ]
    left = feature_frame.copy()
    left[timestamp_col] = pd.to_datetime(left[timestamp_col], errors="coerce", utc=True)
    right = rel[[timestamp_col, *keep_cols]].copy()
    right[timestamp_col] = pd.to_datetime(right[timestamp_col], errors="coerce", utc=True)
    left = left.sort_values(timestamp_col)
    right = right.sort_values(timestamp_col)
    merged = pd.merge_asof(left, right, on=timestamp_col, direction="backward")
    return merged
