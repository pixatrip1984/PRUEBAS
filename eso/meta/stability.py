"""Temporal stability diagnostics for feature-target associations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


STABLE = "STABLE"
MIXED = "MIXED"
UNSTABLE = "UNSTABLE"

_DEFAULT_TARGET = "association"
_GROUP_COL_CANDIDATES = ("dataset_id", "asset_id", "symbol", "horizon", "feature")
_BLOCK_COL_CANDIDATES = ("block", "window", "period", "split", "fold")
_TARGET_COL_CANDIDATES = ("target", "association", "label")
_LONG_CORR_CANDIDATES = ("corr", "correlation", "corr_value", "value")


def score_correlation_stability(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    min_abs_corr: float = 0.05,
    stable_sign_fraction: float = 0.8,
    mixed_sign_fraction: float = 0.6,
    min_blocks: int = 2,
) -> dict[str, Any]:
    """Score one ordered or unordered set of block correlations.

    The score is intentionally conservative. A sequence is stable only when it
    has enough non-null blocks, the dominant sign is consistent, and every
    non-null block clears the minimum absolute-correlation floor.
    """

    _validate_thresholds(min_abs_corr, stable_sign_fraction, mixed_sign_fraction, min_blocks)

    raw = pd.to_numeric(pd.Series(values), errors="coerce")
    arr = raw.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    n_total = int(len(raw))
    n_blocks = int(len(arr))
    n_missing = n_total - n_blocks
    evidence_score = min(1.0, n_blocks / float(min_blocks))

    if n_blocks == 0:
        return {
            "decision": UNSTABLE,
            "stability_score": 0.0,
            "dominant_sign": 0,
            "sign_consistency": 0.0,
            "magnitude_floor_fraction": 0.0,
            "min_abs_corr": np.nan,
            "mean_abs_corr": np.nan,
            "max_abs_corr": np.nan,
            "n_blocks": 0,
            "n_blocks_total": n_total,
            "n_missing": n_missing,
            "required_min_abs_corr": float(min_abs_corr),
        }

    positives = int(np.sum(arr > 0))
    negatives = int(np.sum(arr < 0))
    if positives > negatives:
        dominant_sign = 1
    elif negatives > positives:
        dominant_sign = -1
    else:
        dominant_sign = 0

    sign_consistency = max(positives, negatives) / float(n_blocks)
    abs_values = np.abs(arr)
    min_abs_observed = float(np.min(abs_values))
    mean_abs_corr = float(np.mean(abs_values))
    max_abs_corr = float(np.max(abs_values))
    magnitude_floor_fraction = float(np.mean(abs_values >= min_abs_corr))

    if min_abs_corr == 0:
        magnitude_floor_score = 1.0
    else:
        magnitude_floor_score = min(1.0, min_abs_observed / float(min_abs_corr))
    stability_score = float(sign_consistency * magnitude_floor_score * evidence_score)

    if n_blocks < min_blocks:
        decision = UNSTABLE
    elif max_abs_corr < min_abs_corr:
        decision = UNSTABLE
    elif sign_consistency < mixed_sign_fraction:
        decision = UNSTABLE
    elif sign_consistency >= stable_sign_fraction and min_abs_observed >= min_abs_corr:
        decision = STABLE
    else:
        decision = MIXED

    return {
        "decision": decision,
        "stability_score": stability_score,
        "dominant_sign": dominant_sign,
        "sign_consistency": float(sign_consistency),
        "magnitude_floor_fraction": magnitude_floor_fraction,
        "min_abs_corr": min_abs_observed,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "n_blocks": n_blocks,
        "n_blocks_total": n_total,
        "n_missing": n_missing,
        "required_min_abs_corr": float(min_abs_corr),
    }


def diagnose_association_stability(
    rows: pd.DataFrame | Iterable[dict[str, Any]],
    *,
    group_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    block_col: str | None = None,
    corr_col: str | None = None,
    corr_columns: Sequence[str] | None = None,
    min_abs_corr: float = 0.05,
    stable_sign_fraction: float = 0.8,
    mixed_sign_fraction: float = 0.6,
    min_blocks: int = 2,
) -> pd.DataFrame:
    """Diagnose temporal stability for feature-target correlation rows.

    Accepted inputs:
    - Wide asset-dissection rows such as ``horizon, feature, corr_rv_train,
      corr_rv_test``.
    - Long temporal-block rows such as ``horizon, feature, target, block,
      corr``.
    """

    _validate_thresholds(min_abs_corr, stable_sign_fraction, mixed_sign_fraction, min_blocks)
    df = _as_frame(rows)
    group_cols = _resolve_group_cols(df, group_cols)
    long = _to_long_blocks(
        df,
        group_cols=group_cols,
        target_col=target_col,
        block_col=block_col,
        corr_col=corr_col,
        corr_columns=corr_columns,
    )
    result_columns = [
        *group_cols,
        "target",
        "decision",
        "stability_score",
        "dominant_sign",
        "sign_consistency",
        "magnitude_floor_fraction",
        "min_abs_corr",
        "mean_abs_corr",
        "max_abs_corr",
        "n_blocks",
        "n_blocks_total",
        "n_missing",
        "required_min_abs_corr",
    ]
    if long.empty:
        return pd.DataFrame(columns=result_columns)

    records: list[dict[str, Any]] = []
    by_cols = [*group_cols, "target"]
    for key, group in long.groupby(by_cols, dropna=False, sort=False):
        if len(by_cols) == 1:
            key = (key,)
        record = dict(zip(by_cols, key, strict=True))
        record.update(
            score_correlation_stability(
                group["correlation"],
                min_abs_corr=min_abs_corr,
                stable_sign_fraction=stable_sign_fraction,
                mixed_sign_fraction=mixed_sign_fraction,
                min_blocks=min_blocks,
            )
        )
        records.append(record)

    out = pd.DataFrame(records, columns=result_columns)
    return out.sort_values(by_cols).reset_index(drop=True)


def _validate_thresholds(
    min_abs_corr: float,
    stable_sign_fraction: float,
    mixed_sign_fraction: float,
    min_blocks: int,
) -> None:
    if min_abs_corr < 0:
        raise ValueError("min_abs_corr must be non-negative")
    if min_blocks < 1:
        raise ValueError("min_blocks must be at least 1")
    if not 0 <= mixed_sign_fraction <= 1:
        raise ValueError("mixed_sign_fraction must be between 0 and 1")
    if not 0 <= stable_sign_fraction <= 1:
        raise ValueError("stable_sign_fraction must be between 0 and 1")
    if stable_sign_fraction < mixed_sign_fraction:
        raise ValueError("stable_sign_fraction must be >= mixed_sign_fraction")


def _as_frame(rows: pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _resolve_group_cols(df: pd.DataFrame, group_cols: Sequence[str] | None) -> list[str]:
    if group_cols is None:
        return [col for col in _GROUP_COL_CANDIDATES if col in df.columns]
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"group columns missing from input: {missing}")
    return list(group_cols)


def _to_long_blocks(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_col: str | None,
    block_col: str | None,
    corr_col: str | None,
    corr_columns: Sequence[str] | None,
) -> pd.DataFrame:
    resolved_corr_col = _resolve_named_column(df, corr_col, _LONG_CORR_CANDIDATES)
    if resolved_corr_col is not None:
        return _long_from_long_rows(
            df,
            group_cols=group_cols,
            target_col=_resolve_named_column(df, target_col, _TARGET_COL_CANDIDATES),
            block_col=_resolve_named_column(df, block_col, _BLOCK_COL_CANDIDATES),
            corr_col=resolved_corr_col,
        )

    resolved_corr_columns = list(corr_columns) if corr_columns is not None else [
        col for col in df.columns if col.startswith("corr_")
    ]
    if not resolved_corr_columns:
        raise ValueError("input must contain a long correlation column or corr_* columns")
    missing = [col for col in resolved_corr_columns if col not in df.columns]
    if missing:
        raise ValueError(f"correlation columns missing from input: {missing}")

    return _long_from_wide_rows(
        df,
        group_cols=group_cols,
        target_col=_resolve_named_column(df, target_col, _TARGET_COL_CANDIDATES),
        block_col=_resolve_named_column(df, block_col, _BLOCK_COL_CANDIDATES),
        corr_columns=resolved_corr_columns,
    )


def _resolve_named_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"column missing from input: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _long_from_long_rows(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_col: str | None,
    block_col: str | None,
    corr_col: str,
) -> pd.DataFrame:
    out = df.loc[:, list(group_cols)].copy() if group_cols else pd.DataFrame(index=df.index)
    out["target"] = df[target_col].astype(str) if target_col else _DEFAULT_TARGET
    out["block"] = df[block_col].astype(str) if block_col else df.index.astype(str)
    out["correlation"] = pd.to_numeric(df[corr_col], errors="coerce")
    return out


def _long_from_wide_rows(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_col: str | None,
    block_col: str | None,
    corr_columns: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    duplicated_groups = bool(group_cols) and df.duplicated(list(group_cols), keep=False).any()
    for row_index, row in df.iterrows():
        base = {col: row[col] for col in group_cols}
        row_block = str(row[block_col]) if block_col else None
        row_target = str(row[target_col]) if target_col else None
        for column in corr_columns:
            target, block = _parse_corr_column(column)
            if row_target is not None and target == _DEFAULT_TARGET:
                target = row_target
            if row_block is not None:
                block = f"{row_block}:{block}"
            elif duplicated_groups:
                block = f"{row_index}:{block}"
            records.append({
                **base,
                "target": target,
                "block": block,
                "correlation": row[column],
            })
    out = pd.DataFrame(records)
    if not out.empty:
        out["correlation"] = pd.to_numeric(out["correlation"], errors="coerce")
    return out


def _parse_corr_column(column: str) -> tuple[str, str]:
    if column.startswith("corr_"):
        body = column[len("corr_"):]
    else:
        body = column
    if "_" not in body:
        return _DEFAULT_TARGET, body
    target, block = body.rsplit("_", 1)
    return target or _DEFAULT_TARGET, block or body

