from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DRAW_COLUMNS = [
    "contest", "date", "n1", "n2", "n3", "n4", "n5", "n6", "additional",
    "jackpot", "source", "source_url", "source_hash", "retrieved_at",
]

PRIZE_TIER_COLUMNS = [
    "contest", "tier", "match_natural", "match_additional", "winners",
    "prize_individual", "source", "source_url", "source_hash", "retrieved_at",
    "source_quality",
]


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_contests: int = 0
    notes: tuple[str, ...] = ()


def _coerce_int_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def canonicalize_draws(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "CONCURSO": "contest", "FECHA": "date", "R1": "n1", "R2": "n2",
        "R3": "n3", "R4": "n4", "R5": "n5", "R6": "n6", "R7": "additional",
        "BOLSA": "jackpot",
    }
    out = out.rename(columns=rename)
    required = ["contest", "date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing required draw columns: {missing}")

    out["contest"] = _coerce_int_nullable(out["contest"])
    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "additional"]:
        out[c] = _coerce_int_nullable(out[c])
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=False).dt.date
    out["jackpot"] = pd.to_numeric(out["jackpot"], errors="coerce") if "jackpot" in out.columns else np.nan

    for c in ["source", "source_url", "source_hash", "retrieved_at"]:
        if c not in out.columns:
            out[c] = None
    out = out[[c for c in DRAW_COLUMNS if c in out.columns]]
    return out.sort_values("contest").reset_index(drop=True)


def validate_draws(df: pd.DataFrame, strict: bool = True) -> ValidationSummary:
    x = canonicalize_draws(df)
    nums = x[["n1", "n2", "n3", "n4", "n5", "n6"]].astype("Float64")
    basic = x["contest"].notna() & x["date"].notna() & nums.notna().all(axis=1) & x["additional"].notna()
    in_range = nums.apply(lambda s: s.between(1, 56)).all(axis=1) & x["additional"].between(1, 56)
    arr = nums.to_numpy(dtype=float)
    ordered = pd.Series(np.all(np.diff(arr, axis=1) > 0, axis=1), index=x.index)
    add_not_main = pd.Series([
        int(a) not in {int(v) for v in row if not pd.isna(v)} if not pd.isna(a) else False
        for row, a in zip(arr, x["additional"])
    ], index=x.index)
    valid = basic & in_range & ordered & add_not_main
    dup = int(x["contest"].duplicated().sum())
    notes = []
    if dup:
        notes.append(f"duplicate contest ids: {dup}")
    if strict and (not bool(valid.all()) or dup):
        bad = x.loc[~valid, ["contest", "date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]]
        raise ValueError(f"invalid draws detected: {len(bad)} rows; duplicates={dup}\n{bad.head(10)}")
    return ValidationSummary(len(x), int(valid.sum()), int((~valid).sum()), dup, tuple(notes))


def validate_prize_tiers(df: pd.DataFrame, strict: bool = True) -> ValidationSummary:
    required = ["contest", "tier", "match_natural", "match_additional", "winners", "prize_individual"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required prize-tier columns: {missing}")
    x = df.copy()
    for c in ["contest", "tier", "match_natural", "winners"]:
        x[c] = _coerce_int_nullable(x[c])
    x["match_additional"] = x["match_additional"].astype("boolean")
    x["prize_individual"] = pd.to_numeric(x["prize_individual"], errors="coerce")
    valid = (
        x["contest"].notna() & x["tier"].between(1, 9) & x["match_natural"].between(2, 6)
        & x["winners"].ge(0) & x["prize_individual"].ge(0)
    )
    if strict and not bool(valid.all()):
        raise ValueError(f"invalid prize-tier rows: {int((~valid).sum())}")
    return ValidationSummary(len(x), int(valid.sum()), int((~valid).sum()))
