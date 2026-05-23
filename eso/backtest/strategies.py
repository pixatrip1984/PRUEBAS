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


def proportional_strategy(
    features: pd.DataFrame,
    signal_col: str = "cos_theta_24h",
    confidence_col: str = "ring_radius",
    signal_scale: float = 1.0,
    confidence_percentile_cap: float = 0.95,
) -> pd.Series:
    """Continuous position sizing: signal strength × ring confidence.

    Position at time t = clip(signal[t] * weight[t], -1, 1)

    where weight[t] normalises ring_radius by its rolling 500-bar 95th
    percentile so the position is bounded and comparable across regimes.
    Using ring_radius as a multiplier means the model reduces exposure
    whenever the UMAP embedding drifts off the ring (low-confidence geometry).

    No binary threshold → no discrete jumps → lower turnover than the
    hysteresis strategy, so costs are spread more proportionally to exposure.

    Args:
        features:                 Output of build_feature_vector.
        signal_col:               Directional feature (cos or sin of theta).
        confidence_col:           Confidence weight. ring_radius by default.
        signal_scale:             Multiplier applied to raw signal before clipping.
        confidence_percentile_cap: Cap confidence at this rolling percentile
                                   to avoid extreme scaling in outlier bars.

    Returns:
        Position series in [-1, 1] indexed like features.
    """
    if signal_col not in features.columns:
        raise KeyError(f"signal_col '{signal_col}' not in features.")

    sig = features[signal_col].to_numpy(dtype=float)

    if confidence_col in features.columns:
        conf_raw = features[confidence_col].to_numpy(dtype=float)
        # Rolling percentile cap over 500-bar window
        cap = (
            pd.Series(conf_raw)
            .rolling(500, min_periods=50)
            .quantile(confidence_percentile_cap)
            .to_numpy()
        )
        cap = np.where(cap > 0, cap, np.nanmedian(conf_raw[conf_raw > 0]))
        conf = np.clip(conf_raw / cap, 0.0, 1.0)
    else:
        conf = np.ones(len(sig))

    raw_pos = signal_scale * sig * conf
    desired = np.clip(raw_pos, -1.0, 1.0)
    desired = np.where(np.isnan(desired), 0.0, desired)

    return pd.Series(desired, index=features.index, name="position")


def proportional_deadband_strategy(
    features: pd.DataFrame,
    signal_col: str = "cos_theta_24h",
    confidence_col: str = "ring_radius",
    signal_scale: float = 1.0,
    confidence_percentile_cap: float = 0.95,
    min_trade_size: float = 0.15,
) -> pd.Series:
    """Proportional sizing with a dead-band to suppress micro-rebalancing.

    Identical to proportional_strategy but only updates the held position when
    the desired position differs from the current held position by more than
    `min_trade_size`. This dramatically reduces turnover while keeping most
    of the signal, since small oscillations in phase do not trigger a trade.

    Args:
        min_trade_size: Minimum |Δposition| required to rebalance.
                        At 0.15 a bar must shift position by 15% before
                        incurring any fee. Good range: 0.1–0.3.
    """
    if signal_col not in features.columns:
        raise KeyError(f"signal_col '{signal_col}' not in features.")

    sig = features[signal_col].to_numpy(dtype=float)

    if confidence_col in features.columns:
        conf_raw = features[confidence_col].to_numpy(dtype=float)
        cap = (
            pd.Series(conf_raw)
            .rolling(500, min_periods=50)
            .quantile(confidence_percentile_cap)
            .to_numpy()
        )
        cap = np.where(cap > 0, cap, np.nanmedian(conf_raw[conf_raw > 0]))
        conf = np.clip(conf_raw / cap, 0.0, 1.0)
    else:
        conf = np.ones(len(sig))

    desired = np.clip(signal_scale * sig * conf, -1.0, 1.0)
    desired = np.where(np.isnan(desired), 0.0, desired)

    held = np.zeros(len(desired))
    current = 0.0
    for i in range(len(desired)):
        if abs(desired[i] - current) >= min_trade_size:
            current = desired[i]
        held[i] = current

    return pd.Series(held, index=features.index, name="position")


def gated_phase_threshold_strategy(
    features: pd.DataFrame,
    signal_col: str = "cos_theta_24h",
    gate_col: str = "ring_radius",
    gate_percentile: float = 0.70,
    gate_window: int = 1000,
    enter_threshold: float = 0.3,
    exit_threshold: float = 0.1,
    allow_short: bool = True,
) -> pd.Series:
    """Hysteresis phase threshold AND regime gate.

    Combines the few-trade discipline of phase_threshold_strategy with
    the regime filter of regime_gated_strategy: only allow entries when
    the gate is active. Exits respect the original threshold logic
    regardless of gate state (don't trap a losing position because the
    gate closes).
    """
    if signal_col not in features.columns:
        raise KeyError(f"signal_col '{signal_col}' not in features")
    if gate_col not in features.columns:
        raise KeyError(f"gate_col '{gate_col}' not in features")
    if exit_threshold >= enter_threshold:
        raise ValueError("exit_threshold must be < enter_threshold")

    sig = features[signal_col].to_numpy(dtype=float)
    gate = features[gate_col].to_numpy(dtype=float)
    threshold = (
        pd.Series(gate)
        .rolling(gate_window, min_periods=max(50, gate_window // 10))
        .quantile(gate_percentile)
        .to_numpy()
    )
    active = gate >= threshold

    pos = np.zeros(len(sig), dtype=float)
    state = 0
    for i in range(len(sig)):
        x = sig[i]
        if np.isnan(x):
            pos[i] = state
            continue
        if state == 0:
            # Entries gated
            if active[i]:
                if x >= enter_threshold:
                    state = 1
                elif allow_short and x <= -enter_threshold:
                    state = -1
        elif state == 1:
            # Exits ungated — always allowed to flatten
            if x <= exit_threshold:
                if active[i] and allow_short and x <= -enter_threshold:
                    state = -1
                else:
                    state = 0
        elif state == -1:
            if x >= -exit_threshold:
                if active[i] and x >= enter_threshold:
                    state = 1
                else:
                    state = 0
        pos[i] = state

    return pd.Series(pos, index=features.index, name="position")


def regime_gated_strategy(
    features: pd.DataFrame,
    signal_col: str = "cos_theta_24h",
    gate_col: str = "ring_radius",
    gate_percentile: float = 0.70,
    gate_window: int = 1000,
    signal_scale: float = 1.0,
    min_trade_size: float = 0.15,
    allow_short: bool = True,
) -> pd.Series:
    """Trade the phase signal only when the regime gate is favourable.

    Position = clip(scale × signal, -1, 1) when gate > rolling_percentile(gate_col),
    else 0. Dead-band applied on top.

    The motivation is the stability finding: phase features predict
    direction only when ring_radius is high (geometrically: when the
    embedding is close to the ring). Outside that regime the signal
    has zero or negative correlation with returns and trading it costs
    money.

    Args:
        signal_col:      Directional feature.
        gate_col:        Column used as regime indicator (default ring_radius).
        gate_percentile: Threshold (0-1) above which trading is enabled,
                         computed as a rolling percentile of gate_col.
        gate_window:     Rolling window for the percentile (default 1000 bars).
        signal_scale:    Multiplier on raw signal before clipping.
        min_trade_size:  Dead-band for turnover control.
        allow_short:     If False, gate also blocks shorts.

    Returns:
        Position series in [-1, 1].
    """
    if signal_col not in features.columns:
        raise KeyError(f"signal_col '{signal_col}' not in features")
    if gate_col not in features.columns:
        raise KeyError(f"gate_col '{gate_col}' not in features")
    if not 0.0 < gate_percentile < 1.0:
        raise ValueError("gate_percentile must be in (0, 1)")

    sig = features[signal_col].to_numpy(dtype=float)
    gate = features[gate_col].to_numpy(dtype=float)

    # Rolling percentile threshold of the gate column itself
    threshold = (
        pd.Series(gate)
        .rolling(gate_window, min_periods=max(50, gate_window // 10))
        .quantile(gate_percentile)
        .to_numpy()
    )
    active = gate >= threshold

    raw = signal_scale * sig
    desired = np.clip(raw, -1.0, 1.0)
    if not allow_short:
        desired = np.clip(desired, 0.0, 1.0)
    desired = np.where(np.isnan(desired) | ~active, 0.0, desired)

    held = np.zeros(len(desired))
    current = 0.0
    for i in range(len(desired)):
        if abs(desired[i] - current) >= min_trade_size:
            current = desired[i]
        held[i] = current

    return pd.Series(held, index=features.index, name="position")
