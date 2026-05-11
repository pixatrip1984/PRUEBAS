# btc_dataset_builder.py

import os
import zipfile
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
from tqdm import tqdm


@dataclass
class BinanceKlineConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"

    start_year: int = 2017
    start_month: int = 8

    end_year: int = 2026
    end_month: int = 4

    output_dir: str = "./data/btc"
    delete_zips_after_reading: bool = True

    save_csv: bool = False
    save_parquet: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class BinanceBTCDatasetBuilder:
    """
    Construye un dataset crudo de BTCUSDT para entrenamiento de redes neuronales.

    No calcula indicadores técnicos.
    Solo descarga y normaliza datos crudos de klines:

    - open_time
    - open
    - high
    - low
    - close
    - volume
    - close_time
    - quote_asset_volume
    - number_of_trades
    - taker_buy_base_asset_volume
    - taker_buy_quote_asset_volume
    - ignore

    Además agrega:
    - open_datetime
    - close_datetime
    - open_time_ms
    - close_time_ms
    """

    COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    FLOAT_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    def __init__(self, config: BinanceKlineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.zip_dir = self.output_dir / "zips"
        self.raw_dir = self.output_dir / "raw"
        self.splits_dir = self.output_dir / "splits"

        self.zip_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> pd.DataFrame:
        dfs = []

        for year, month in self._month_range():
            filename = self._monthly_filename(year, month)
            zip_path = self.zip_dir / filename
            url = self._monthly_url(year, month)

            if not zip_path.exists():
                ok = self._download_file(url, zip_path)
                if not ok:
                    print(f"No disponible: {filename}")
                    continue

            try:
                df_month = self._read_zip_csv(zip_path)
                dfs.append(df_month)
                print(f"OK: {filename} | filas: {len(df_month):,}")

                if self.config.delete_zips_after_reading:
                    zip_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"Error leyendo {filename}: {e}")

        if not dfs:
            raise RuntimeError("No se descargó ningún archivo válido.")

        df = pd.concat(dfs, ignore_index=True)
        df = self._clean_and_normalize(df)
        self._save_full_dataset(df)
        self._save_temporal_splits(df)

        return df

    def _month_range(self):
        y = self.config.start_year
        m = self.config.start_month

        while (y < self.config.end_year) or (
            y == self.config.end_year and m <= self.config.end_month
        ):
            yield y, m
            m += 1
            if m == 13:
                y += 1
                m = 1

    def _monthly_filename(self, year: int, month: int) -> str:
        return f"{self.config.symbol}-{self.config.interval}-{year}-{month:02d}.zip"

    def _monthly_url(self, year: int, month: int) -> str:
        filename = self._monthly_filename(year, month)

        return (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"{self.config.symbol}/{self.config.interval}/{filename}"
        )

    def _download_file(self, url: str, path: Path) -> bool:
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code != 200:
            return False

        total = int(response.headers.get("content-length", 0))

        with open(path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=path.name,
            leave=False,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    def _read_zip_csv(self, zip_path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None)

        df = df.iloc[:, :12]
        df.columns = self.COLUMNS

        return df

    def _clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.FLOAT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

        df["number_of_trades"] = (
            pd.to_numeric(df["number_of_trades"], errors="coerce")
            .astype("Int64")
        )

        df["open_datetime"] = self._normalize_binance_timestamp(df["open_time"])
        df["close_datetime"] = self._normalize_binance_timestamp(df["close_time"])

        df = df.dropna(subset=["open_datetime", "close_datetime"])

        df["open_time_ms"] = (
            df["open_datetime"].astype("int64") // 1_000_000
        ).astype("Int64")

        df["close_time_ms"] = (
            df["close_datetime"].astype("int64") // 1_000_000
        ).astype("Int64")

        df = df.sort_values("open_datetime")
        df = df.drop_duplicates(subset=["open_datetime"])
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _normalize_binance_timestamp(series: pd.Series) -> pd.Series:
        """
        Binance puede tener timestamps en milisegundos o microsegundos.

        ms actual aprox: 1.7e12
        us actual aprox: 1.7e15
        """
        s = pd.to_numeric(series, errors="coerce")

        is_micro = s > 100_000_000_000_000

        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

        out.loc[~is_micro] = pd.to_datetime(
            s.loc[~is_micro],
            unit="ms",
            utc=True,
            errors="coerce",
        )

        out.loc[is_micro] = pd.to_datetime(
            s.loc[is_micro],
            unit="us",
            utc=True,
            errors="coerce",
        )

        return out

    def _save_full_dataset(self, df: pd.DataFrame):
        base_name = f"{self.config.symbol}_{self.config.interval}_raw_klines"

        if self.config.save_parquet:
            path = self.raw_dir / f"{base_name}.parquet"
            df.to_parquet(path, index=False)
            print(f"Dataset completo guardado en: {path}")

        if self.config.save_csv:
            path = self.raw_dir / f"{base_name}.csv"
            df.to_csv(path, index=False)
            print(f"Dataset completo guardado en: {path}")

    def _save_temporal_splits(self, df: pd.DataFrame):
        train_df, val_df, test_df = self.temporal_split(
            df,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )

        train_path = self.splits_dir / "train.parquet"
        val_path = self.splits_dir / "val.parquet"
        test_path = self.splits_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        print("\nSplits temporales:")
        print(f"Train: {len(train_df):,} filas | {train_df['open_datetime'].min()} -> {train_df['open_datetime'].max()}")
        print(f"Val:   {len(val_df):,} filas | {val_df['open_datetime'].min()} -> {val_df['open_datetime'].max()}")
        print(f"Test:  {len(test_df):,} filas | {test_df['open_datetime'].min()} -> {test_df['open_datetime'].max()}")

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        total = train_ratio + val_ratio + test_ratio

        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio debe ser igual a 1.0")

        n = len(df)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df


class NeuralMarketDataset:
    """
    Wrapper simple para preparar ventanas secuenciales.

    No calcula indicadores.
    Solo transforma columnas crudas en secuencias X/y.

    Ejemplo:
    X[t] = últimas N velas
    y[t] = close futuro, retorno futuro, dirección futura, etc.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "close",
        lookback: int = 256,
        horizon: int = 1,
        target_mode: str = "future_return",
    ):
        self.df = df.copy()
        self.lookback = lookback
        self.horizon = horizon
        self.target_column = target_column
        self.target_mode = target_mode

        if feature_columns is None:
            self.feature_columns = [
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
        else:
            self.feature_columns = feature_columns

        self.df = self.df.dropna(subset=self.feature_columns + [target_column])
        self.df = self.df.reset_index(drop=True)

    def to_numpy(self):
        import numpy as np

        features = self.df[self.feature_columns].astype("float32").values
        target = self.df[self.target_column].astype("float32").values

        X = []
        y = []

        max_i = len(self.df) - self.horizon

        for i in range(self.lookback, max_i):
            x_window = features[i - self.lookback:i]

            current_price = target[i - 1]
            future_price = target[i + self.horizon - 1]

            if self.target_mode == "future_price":
                y_value = future_price

            elif self.target_mode == "future_return":
                y_value = (future_price / current_price) - 1.0

            elif self.target_mode == "direction":
                y_value = 1.0 if future_price > current_price else 0.0

            else:
                raise ValueError(
                    "target_mode debe ser: future_price, future_return o direction"
                )

            X.append(x_window)
            y.append(y_value)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        return X, y