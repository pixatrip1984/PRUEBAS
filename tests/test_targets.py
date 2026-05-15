import numpy as np
import pandas as pd
import pytest

from eso.signals.targets import (
    build_market_target_frame,
    future_direction,
    future_log_return,
    future_realized_volatility,
    regime_from_training_median,
)


def test_future_log_return_indexes_decision_time():
    close = pd.Series([100.0, 110.0, 121.0, 133.1], index=[10, 11, 12, 13])
    y = future_log_return(close, horizon=2)
    assert y.index.tolist() == [10, 11, 12, 13]
    assert np.isclose(y.loc[10], np.log(121.0 / 100.0))
    assert np.isnan(y.loc[12])
    assert np.isnan(y.loc[13])


def test_future_realized_volatility_uses_next_returns_only():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    y = future_realized_volatility(close, horizon=2)
    assert np.isclose(y.iloc[0], 0.0)
    assert np.isclose(y.iloc[1], 0.0)
    assert np.isnan(y.iloc[2])


def test_horizon_one_realized_volatility_is_abs_next_return():
    close = pd.Series([100.0, 90.0, 99.0])
    y = future_realized_volatility(close, horizon=1)
    assert np.isclose(y.iloc[0], abs(np.log(90.0 / 100.0)))
    assert np.isnan(y.iloc[-1])


def test_future_direction_can_mark_flat_as_nan():
    close = pd.Series([100.0, 100.0, 101.0])
    y = future_direction(close, horizon=1, drop_flat=True)
    assert np.isnan(y.iloc[0])
    assert y.iloc[1] == 1


def test_regime_from_training_median_uses_training_slice_only():
    target = pd.Series([1.0, 3.0, 100.0, 200.0, np.nan])
    y, threshold = regime_from_training_median(target, train_size=2)
    assert threshold == 2.0
    assert y.iloc[0] == 0
    assert y.iloc[1] == 1
    assert y.iloc[2] == 1
    assert np.isnan(y.iloc[4])


def test_build_market_target_frame_has_all_requested_horizons():
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    targets = build_market_target_frame(df, horizons=(1, 3))
    assert "future_lr_1" in targets.columns
    assert "future_direction_3" in targets.columns
    assert "future_rv_3" in targets.columns


def test_invalid_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        future_log_return(pd.Series([1.0, 2.0]), horizon=0)
