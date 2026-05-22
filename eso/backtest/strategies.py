"""Baseline strategies on top of the 9-feature causal vector.

These are intentionally simple. The goal is to measure the *signal*, not to
beat the market with engineering. Anything that depends on parameters fit on
the same window it's tested on is forbidden here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_only_baseline(features: pd.DataFrame) -> pd.Series:
    """Always long. Reference cost of buy-and-hold over the same window."""
    return pd.Series(1.0, index=features.index, name="position")


def phase_threshold_strategy(
    features: pd.DataFrame,
    signal_col: str = "cos_theta_24h",
    enter_threshold: float = 0.3,
    exit_threshold: float = 0.1,
    allow_short: bool = True,
) -> pd.Series:
    """Hysteresis strategy on a single causal phase coordinate.

    Goes long when signal_col rises above +enter_threshold, exits when it
    falls below +exit_threshold. Symmetric for short side.

    Hysteresis (enter > exit) is the cheap way to control turnover without
    fitting any parameter to the data.

    Args:
        features:         Output of build_feature_vector or CausalFeatureBuilder.
        signal_col:       Column to use as primary signal. cos_theta_24h is a
                          good default — medium-cycle position, OOS-validated.
        enter_threshold:  Absolute value to trigger entry (in feature units).
        exit_threshold:   Absolute value to trigger exit. Must be < enter.
        allow_short:      If False, only long/flat positions.

    Returns:
        Position series in {-1, 0, +1} indexed like features.
    """
    if signal_col not in features.columns:
        raise KeyError(f"signal_col '{signal_col}' not in features. "
                       f"Available: {list(features.columns)}")
    if exit_threshold >= enter_threshold:
        raise ValueError("exit_threshold must be strictly less than enter_threshold")

    sig = features[signal_col].to_numpy(dtype=float)
    pos = np.zeros(len(sig), dtype=float)
    state = 0  # -1, 0, +1

    for i in range(len(sig)):
        x = sig[i]
        if np.isnan(x):
            pos[i] = state
            continue
        if state == 0:
            if x >= enter_threshold:
                state = 1
            elif allow_short and x <= -enter_threshold:
                state = -1
        elif state == 1:
            if x <= exit_threshold:
                # Flip directly to short if signal crossed the other way fully
                if allow_short and x <= -enter_threshold:
                    state = -1
                else:
                    state = 0
        elif state == -1:
            if x >= -exit_threshold:
                if x >= enter_threshold:
                    state = 1
                else:
                    state = 0
        pos[i] = state

    return pd.Series(pos, index=features.index, name="position")
