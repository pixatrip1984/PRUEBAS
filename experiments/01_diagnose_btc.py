"""Run ESO blind diagnosis on a simple dataset.

The script defaults to a synthetic harmonic oscillator so it can run without
external data. Use --csv or --parquet later for real datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eso.diagnostics.report import run_diagnosis


def harmonic(n: int = 1000):
    t = np.linspace(0.0, 8.0 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def load_numeric(path: str):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.select_dtypes(include=["number"]).to_numpy(dtype=float)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.select_dtypes(include=["number"]).to_numpy(dtype=float)
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    data = load_numeric(args.path) if args.path else harmonic(args.n)
    report = run_diagnosis(data)
    print(report["summary"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
