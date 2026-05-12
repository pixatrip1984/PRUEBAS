"""Data loading helpers for ESO."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_numeric(path: str, columns: list[str] | None = None) -> np.ndarray:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        arr = np.load(path)
        return np.asarray(arr, dtype=float)
    if columns is not None:
        df = df[columns]
    return df.select_dtypes(include=["number"]).to_numpy(dtype=float)


def make_windows(data, lookback: int, step: int = 1) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if len(x) < lookback:
        raise ValueError("data length must be >= lookback")
    return np.stack([x[i - lookback : i] for i in range(lookback, len(x) + 1, step)], axis=0)
