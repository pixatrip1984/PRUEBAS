"""
BTCDirectionDatasetPlain — Dataset sin SHSE para el baseline.

Devuelve (x, y) tensores en lugar de grafos PyG.
Usa exactamente la misma normalización IQR que SHSEEncoder
para que la comparación sea justa (solo cambia la arquitectura).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


class BTCDirectionDatasetPlain(Dataset):
    """
    Mismos parámetros que BTCDirectionDataset pero sin codificación SHSE.
    x.shape = (lookback, n_features)
    y = scalar long (0=baja, 1=sube)
    """

    def __init__(
        self,
        parquet_path: str,
        lookback: int = 64,
        horizon: int = 5,
        step: int = 10,
        max_samples: int = None,
    ):
        df = pd.read_parquet(parquet_path)
        self.features = df[FEATURE_COLS].astype(np.float32).values
        self.close = df["close"].astype(np.float32).values
        self.lookback = lookback
        self.horizon = horizon

        max_i = len(df) - horizon
        indices = list(range(lookback, max_i, step))
        if max_samples:
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        window = self.features[i - self.lookback : i]  # (lookback, 9)

        # Misma normalización IQR que SHSEEncoder
        q1 = np.percentile(window, 25, axis=0)
        q3 = np.percentile(window, 75, axis=0)
        scale = np.maximum(q3 - q1, 1e-6)
        median = np.median(window, axis=0)
        normed = np.clip((window - median) / scale, -4.0, 4.0).astype(np.float32)

        label = 1 if self.close[i + self.horizon - 1] > self.close[i - 1] else 0
        return torch.from_numpy(normed), torch.tensor(label, dtype=torch.long)
