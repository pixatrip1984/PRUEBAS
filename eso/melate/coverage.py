from __future__ import annotations

import pandas as pd


def coverage_report(draws: pd.DataFrame, prize_tiers: pd.DataFrame) -> pd.DataFrame:
    d = draws.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["year"] = d["date"].dt.year
    complete = set()
    if not prize_tiers.empty:
        counts = prize_tiers.groupby("contest")["tier"].nunique()
        complete = set(counts[counts >= 9].index.astype(int))
    d["has_full_9_tiers"] = d["contest"].astype(int).isin(complete)
    out = d.groupby("year", dropna=False).agg(
        draws=("contest", "count"),
        full_prize_tiers=("has_full_9_tiers", "sum"),
    ).reset_index()
    out["coverage"] = out["full_prize_tiers"] / out["draws"]
    return out
