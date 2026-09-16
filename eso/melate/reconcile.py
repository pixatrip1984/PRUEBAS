from __future__ import annotations

import pandas as pd

from .schema import canonicalize_draws, validate_prize_tiers


def reconcile_prize_tiers(
    official_draws: pd.DataFrame,
    scraped_meta: pd.DataFrame,
    prize_tiers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept secondary prize rows only when draw identity matches official data."""
    draws = canonicalize_draws(official_draws)
    off = draws.set_index("contest")
    accepted_contests: set[int] = set()
    quarantine_reasons = []

    for _, row in scraped_meta.iterrows():
        c = int(row["contest"])
        if c not in off.index:
            quarantine_reasons.append({"contest": c, "reason": "contest_missing_from_official"})
            continue
        d = off.loc[c]
        expected = tuple(int(d[f"n{i}"]) for i in range(1, 7))
        got = tuple(int(x) for x in row["main"])
        if expected != got:
            quarantine_reasons.append({"contest": c, "reason": f"main_mismatch official={expected} scraped={got}"})
            continue
        if int(d["additional"]) != int(row["additional"]):
            quarantine_reasons.append({"contest": c, "reason": f"additional_mismatch official={int(d['additional'])} scraped={int(row['additional'])}"})
            continue
        accepted_contests.add(c)

    accepted = prize_tiers[prize_tiers["contest"].isin(accepted_contests)].copy()
    if not accepted.empty:
        accepted["source_quality"] = 2
        validate_prize_tiers(accepted, strict=True)

    q_contests = {int(x["contest"]) for x in quarantine_reasons}
    quarantine = prize_tiers[prize_tiers["contest"].isin(q_contests)].copy()
    if quarantine_reasons:
        reason_df = pd.DataFrame(quarantine_reasons).drop_duplicates("contest")
        quarantine = quarantine.merge(reason_df, on="contest", how="right")
    return accepted.reset_index(drop=True), quarantine.reset_index(drop=True)
