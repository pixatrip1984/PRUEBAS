"""Run ESO blind diagnosis on synthetic or real numeric data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eso.data.loader import load_numeric
from eso.data.synthetic import generate_dataset
from eso.diagnostics.report import run_diagnosis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--synthetic", type=str, default="harmonic")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    data = load_numeric(args.path) if args.path else generate_dataset(args.synthetic, n=args.n)
    report = run_diagnosis(data)
    print(report["summary"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
