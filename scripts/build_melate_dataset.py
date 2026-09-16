from __future__ import annotations

import argparse
from pathlib import Path

from eso.melate.sources.loteria_nacional import fetch_official_history, load_official_file
from eso.melate.sources.melate_e import MelateEClient
from eso.melate.reconcile import reconcile_prize_tiers
from eso.melate.dataset import write_canonical_dataset


def main():
    p = argparse.ArgumentParser(description="Build canonical Melate dataset")
    p.add_argument("--output", default="data/melate/canonical")
    p.add_argument("--raw", default="data/melate/raw")
    p.add_argument("--official-file", help="Use an already-downloaded official Melate CSV")
    p.add_argument("--from-contest", type=int, default=None)
    p.add_argument("--to-contest", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.7)
    args = p.parse_args()

    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    if args.official_file:
        draws = load_official_file(args.official_file)
    else:
        draws = fetch_official_history(raw_dir=raw / "official")

    start = args.from_contest or int(draws["contest"].min())
    end = args.to_contest or int(draws["contest"].max())
    client = MelateEClient(delay_seconds=args.delay, raw_dir=raw / "melate_e")
    meta, tiers, errors = client.fetch_range(start, end, continue_on_error=True)
    accepted, quarantine = reconcile_prize_tiers(draws, meta, tiers)
    if not errors.empty:
        errors.to_csv(raw / "fetch_errors.csv", index=False)
    manifest = write_canonical_dataset(args.output, draws, accepted, quarantine)
    print(manifest)


if __name__ == "__main__":
    main()
