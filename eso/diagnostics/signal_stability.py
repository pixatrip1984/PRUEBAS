"""Rolling stability analysis of cycle-phase predictive power.

Measures the correlation between each phase feature and future log-returns
over rolling windows. The motivating finding was that the full-OOS
correlation r=0.32 averages over sub-regimes where the relationship
reverses sign — this module surfaces those regimes.

Outputs let us answer two questions:
  1. Is the signal stationary? (rolling corr roughly constant vs flipping)
  2. If not, what observable predicts the regime? (cross-correlation of
     rolling-corr against vol_20, ring_radius, lr_z20, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StabilityReport:
    rolling_corr: pd.DataFrame   # cols: each phase feature × rolling correlation with target
    target_horizon: int
    window: int
    regime_indicators: pd.DataFrame  # rolling vol_20, ring_radius, etc., aligned
    flip_count: dict                  # n sign flips per feature
    mean_corr: dict                   # mean rolling corr per feature
    cross_corr_with_regime: dict      # corr(rolling_corr, regime_indicator) per pair
    summary: str = ""


def _rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation between two equal-length series."""
    s_a = pd.Series(a)
    s_b = pd.Series(b)
    return s_a.rolling(window).corr(s_b).to_numpy()


def _sign_flips(x: np.ndarray) -> int:
    """Count sign changes ignoring NaN and zeros."""
    valid = x[~np.isnan(x)]
    if len(valid) < 2:
        return 0
    s = np.sign(valid)
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def analyse_signal_stability(
    features: pd.DataFrame,
    horizon: int = 12,
    window: int = 1000,
    phase_cols: list[str] | None = None,
    regime_cols: list[str] | None = None,
) -> StabilityReport:
    """Compute rolling correlation of each phase feature with future return.

    Args:
        features:   build_feature_vector output (must contain 'close').
        horizon:    Forward horizon in bars for the target return.
        window:     Rolling window size in bars (default 1000 ≈ 42 days at 1h).
        phase_cols: Phase features to test. Defaults to all 6 sin/cos columns.
        regime_cols: Candidate regime indicators. Defaults to vol_20, ring_radius, lr_z20.

    Returns:
        StabilityReport with per-feature rolling correlations and regime cross-corrs.
    """
    if "close" not in features.columns:
        raise KeyError("features must contain 'close'")

    phase_cols = phase_cols or [
        "sin_theta_6h", "cos_theta_6h",
        "sin_theta_24h", "cos_theta_24h",
        "sin_theta_72h", "cos_theta_72h",
    ]
    phase_cols = [c for c in phase_cols if c in features.columns]
    regime_cols = regime_cols or ["vol_20", "ring_radius", "lr_z20"]
    regime_cols = [c for c in regime_cols if c in features.columns]

    log_close = np.log(features["close"].to_numpy(dtype=float))
    target = np.full(len(features), np.nan)
    target[:-horizon] = log_close[horizon:] - log_close[:-horizon]

    roll_data = {}
    flips = {}
    means = {}
    for c in phase_cols:
        rc = _rolling_corr(features[c].to_numpy(dtype=float), target, window)
        roll_data[c] = rc
        flips[c] = _sign_flips(rc)
        valid = rc[~np.isnan(rc)]
        means[c] = float(valid.mean()) if len(valid) else float("nan")

    rolling_corr = pd.DataFrame(roll_data, index=features.index)

    # Regime indicators rolled (rolling mean of each) for visualisation alignment
    regime_data = {}
    for c in regime_cols:
        if c in features.columns:
            regime_data[c] = pd.Series(features[c].to_numpy(dtype=float)).rolling(window).mean().to_numpy()
    regime_indicators = pd.DataFrame(regime_data, index=features.index)

    # Cross-corr between each rolling-corr trace and each regime indicator
    cross = {}
    for pc in phase_cols:
        rc = rolling_corr[pc].to_numpy()
        for ri in regime_cols:
            if ri not in regime_indicators.columns:
                continue
            ri_v = regime_indicators[ri].to_numpy()
            valid = ~(np.isnan(rc) | np.isnan(ri_v))
            if valid.sum() < 50:
                cross[f"{pc}__{ri}"] = float("nan")
                continue
            cross[f"{pc}__{ri}"] = float(np.corrcoef(rc[valid], ri_v[valid])[0, 1])

    # Summary
    most_flips = max(flips, key=flips.get) if flips else None
    least_flips = min(flips, key=flips.get) if flips else None
    summary_lines = [
        f"Rolling stability: window={window}, horizon={horizon}, "
        f"n_bars={len(features)}",
        "",
        "Per-feature rolling correlation with future log-return:",
    ]
    for c in phase_cols:
        summary_lines.append(
            f"  {c:18s} mean={means[c]:+.3f}   sign_flips={flips[c]}"
        )
    summary_lines.append("")
    if cross:
        top = sorted(cross.items(), key=lambda kv: -abs(kv[1]) if not np.isnan(kv[1]) else 0)
        summary_lines.append("Top regime-indicator cross-correlations (|r|):")
        for k, v in top[:8]:
            summary_lines.append(f"  {k:42s} r={v:+.3f}")
    summary_lines.append("")
    if most_flips and least_flips:
        summary_lines.append(
            f"Most unstable: {most_flips} ({flips[most_flips]} flips). "
            f"Most stable: {least_flips} ({flips[least_flips]} flips)."
        )

    return StabilityReport(
        rolling_corr=rolling_corr,
        target_horizon=horizon,
        window=window,
        regime_indicators=regime_indicators,
        flip_count=flips,
        mean_corr=means,
        cross_corr_with_regime=cross,
        summary="\n".join(summary_lines),
    )
