"""Data loading helpers for ESO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import flatten_windows, normalize
from .validation import DataValidationReport, validate_dataframe


@dataclass
class LoadedDataset:
    data: np.ndarray
    validation: DataValidationReport
    source_path: str
    columns: list[str]
    normalized: str
    window_size: int | None = None
    window_step: int | None = None
    window_mode: str | None = None

    def info(self) -> dict:
        return {
            "source_path": self.source_path,
            "columns": self.columns,
            "shape": list(self.data.shape),
            "normalized": self.normalized,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_mode": self.window_mode,
            "validation": self.validation.to_dict(),
        }


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            first_key = sorted(arr.files)[0]
            arr = arr[first_key]
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
    raise ValueError(f"unsupported file extension: {suffix}")


def load_numeric(path: str, columns: list[str] | None = None) -> np.ndarray:
    df = read_table(path)
    report = validate_dataframe(df, columns=columns)
    return df[report.selected_columns].to_numpy(dtype=float)


def make_windows(data, lookback: int, step: int = 1) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if len(x) < lookback:
        raise ValueError("data length must be >= lookback")
    return np.stack([x[i - lookback : i] for i in range(lookback, len(x) + 1, step)], axis=0)


def load_dataset(
    path: str,
    columns: list[str] | None = None,
    normalize_method: str = "robust",
    max_rows: int | None = None,
    window_size: int | None = None,
    window_step: int = 1,
    window_mode: str = "last",
) -> LoadedDataset:
    df = read_table(path)
    if max_rows is not None:
        df = df.head(int(max_rows))
    validation = validate_dataframe(df, columns=columns)
    selected = validation.recommended_columns
    data = df[selected].to_numpy(dtype=float)
    data = normalize(data, method=normalize_method)
    if window_size:
        windows = make_windows(data, lookback=int(window_size), step=int(window_step))
        data = flatten_windows(windows, mode=window_mode)
    return LoadedDataset(
        data=data,
        validation=validation,
        source_path=path,
        columns=selected,
        normalized=normalize_method,
        window_size=window_size,
        window_step=window_step if window_size else None,
        window_mode=window_mode if window_size else None,
    )
