"""Causal target builders for downstream market models.

All labels are indexed at decision time t. A value at row t may use future
prices to define the supervised target, but no feature at row t should use
those future prices. Keeping target construction here avoids repeating the
off-by-one bugs that are easy to introduce in experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return horizon


def _close_series(data: pd.DataFrame | pd.Series | np.ndarray, close_col: str = "close") -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if close_col not in data.columns:
            raise ValueError(f"close column not found: {close_col}")
        return pd.to_numeric(data[close_col], errors="coerce")
    if isinstance(data, pd.Series):
        return pd.to_numeric(data, errors="coerce")
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("array input must be 1-dimensional close prices")
    return pd.Series(arr)


def future_log_return(
    data: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int = 12,
    close_col: str = "close",
) -> pd.Series:
    """Return log(close[t+horizon] / close[t]) indexed at time t."""
    horizon = _validate_horizon(horizon)
    close = _close_series(data, close_col=close_col).astype(float)
    log_close = np.log(close.replace(0, np.nan))
    out = log_close.shift(-horizon) - log_close
    out.name = f"future_lr_{horizon}"
    return out


def future_direction(
    data: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int = 12,
    close_col: str = "close",
    drop_flat: bool = False,
) -> pd.Series:
    """Return sign of future_log_return in {-1, 0, +1}, indexed at time t."""
    lr = future_log_return(data, horizon=horizon, close_col=close_col)
    out = np.sign(lr)
    out = pd.Series(out, index=lr.index, name=f"future_direction_{horizon}")
    if drop_flat:
        out = out.where(out != 0)
    return out


def future_realized_volatility(
    data: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int = 12,
    close_col: str = "close",
    ddof: int = 1,
) -> pd.Series:
    """Return realized volatility of the next `horizon` one-bar returns.

    For horizon=1, absolute next-bar log return is returned because a sample
    standard deviation over one value is undefined.
    """
    horizon = _validate_horizon(horizon)
    close = _close_series(data, close_col=close_col).astype(float)
    log_close = np.log(close.replace(0, np.nan))
    forward_ret = log_close.shift(-1) - log_close

    if horizon == 1:
        out = forward_ret.abs()
    else:
        out = (
            forward_ret.iloc[::-1]
            .rolling(horizon, min_periods=horizon)
            .std(ddof=ddof)
            .iloc[::-1]
        )
    out.name = f"future_rv_{horizon}"
    return out


def regime_from_training_median(
    target: pd.Series,
    train_size: int,
    high_label: int = 1,
    low_label: int = 0,
) -> tuple[pd.Series, float]:
    """Binarize a continuous target using the median of the training window."""
    train_size = int(train_size)
    if train_size < 1:
        raise ValueError("train_size must be >= 1")
    clean_train = target.dropna().iloc[:train_size]
    if clean_train.empty:
        raise ValueError("target has no non-null training values")
    threshold = float(clean_train.median())
    out = pd.Series(np.nan, index=target.index, name=f"{target.name}_regime")
    mask = target.notna()
    out.loc[mask] = np.where(target.loc[mask] > threshold, high_label, low_label)
    return out, threshold


def build_market_target_frame(
    data: pd.DataFrame | pd.Series | np.ndarray,
    horizons: tuple[int, ...] = (12,),
    close_col: str = "close",
) -> pd.DataFrame:
    """Build common market targets for one or more horizons."""
    close = _close_series(data, close_col=close_col)
    out = pd.DataFrame(index=close.index)
    for horizon in horizons:
        h = _validate_horizon(horizon)
        out[f"future_lr_{h}"] = future_log_return(close, horizon=h)
        out[f"future_direction_{h}"] = future_direction(close, horizon=h)
        out[f"future_rv_{h}"] = future_realized_volatility(close, horizon=h)
    return out
