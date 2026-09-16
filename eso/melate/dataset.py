from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .schema import canonicalize_draws, validate_draws, validate_prize_tiers
from .coverage import coverage_report


def write_canonical_dataset(
    output_dir: str | Path,
    draws: pd.DataFrame,
    prize_tiers: pd.DataFrame,
    quarantine: pd.DataFrame | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    draws = canonicalize_draws(draws)
    validate_draws(draws, strict=True)
    if not prize_tiers.empty:
        validate_prize_tiers(prize_tiers, strict=True)

    draws.to_parquet(out / "draws.parquet", index=False)
    draws.to_csv(out / "draws.csv", index=False)
    prize_tiers.to_parquet(out / "prize_tiers.parquet", index=False)
    prize_tiers.to_csv(out / "prize_tiers.csv", index=False)
    if quarantine is not None and not quarantine.empty:
        quarantine.to_csv(out / "quarantine.csv", index=False)

    cov = coverage_report(draws, prize_tiers)
    cov.to_csv(out / "coverage.csv", index=False)
    manifest = {
        "draws": int(len(draws)),
        "contest_min": int(draws["contest"].min()),
        "contest_max": int(draws["contest"].max()),
        "prize_tier_rows": int(len(prize_tiers)),
        "contests_with_prize_tiers": int(prize_tiers["contest"].nunique()) if not prize_tiers.empty else 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
