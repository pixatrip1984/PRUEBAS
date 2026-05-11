"""
BTCDirectionDataset — PyG dataset para predicción de dirección BTC.

Cada muestra es un grafo SHSE que cubre `lookback` minutos.
Label = 1 si close[t + horizon] > close[t - 1], sino 0.
Lazy: los grafos se construyen en __getitem__, no se pre-cargan.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from shse_encoder import SHSEEncoder

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


class BTCDirectionDataset(Dataset):
    """
    Sliding-window BTC direction dataset backed by a parquet split.

    Parameters
    ----------
    parquet_path : str
        Path to train/val/test.parquet.
    lookback : int
        Minutes of history per sample.
    horizon : int
        Minutes ahead to measure direction.
    step : int
        Stride between samples (use > 1 to subsample; avoids near-duplicate windows).
    encoder : SHSEEncoder
        Shared encoder instance (reuses pre-computed sphere geometry).
    max_samples : int | None
        Cap total samples (useful for CPU training).
    """

    def __init__(
        self,
        parquet_path: str,
        lookback: int = 64,
        horizon: int = 5,
        step: int = 30,
        encoder: SHSEEncoder = None,
        max_samples: int = None,
    ):
        df = pd.read_parquet(parquet_path)
        self.features = df[FEATURE_COLS].astype(np.float32).values
        self.close = df["close"].astype(np.float32).values
        self.encoder = encoder or SHSEEncoder(lookback=lookback)
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
        window = self.features[i - self.lookback : i]
        label = 1 if self.close[i + self.horizon - 1] > self.close[i - 1] else 0
        return self.encoder.encode(window, label)
