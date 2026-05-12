"""Cycle phase extraction from the non-linear ring structure in BTC features.

The ring (S¹) found in BTC compact features under UMAP-2D embedding is
quantified here as a continuous phase signal θ(t) ∈ [-π, π].

Pipeline:
  raw OHLCV+microstructure
    → build_financial_features()      compact 4D stationary features
    → UMAP(n_components=2)            non-linear 2D embedding
    → arctan2(y, x)                   phase angle θ
    → unwrap / smooth                 continuous phase (optional)
    → output: pd.Series with timestamps

Usage:
    from eso.signals.cycle import extract_cycle_phase, analyse_cycle_period

    df = pd.read_csv('data/BTCUSDT_1h.csv')
    phase = extract_cycle_phase(df, n_neighbors=15, seed=42)
    period_info = analyse_cycle_period(phase, bars_per_day=24)
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from eso.data.features import build_financial_features, feature_column_groups
from eso.data.preprocess import normalize


def _embed_umap_2d(
    data: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Return UMAP-2D embedding of normalised compact features."""
    try:
        import umap as umap_lib
    except ImportError as e:
        raise ImportError("umap-learn required: pip install umap-learn") from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
            verbose=False,
        )
        return reducer.fit_transform(data)


def extract_cycle_phase(
    df: pd.DataFrame,
    feature_mode: Literal["compact", "returns_vol", "full"] = "compact",
    normalize_method: str = "robust",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    timestamp_col: str | None = "timestamp",
    smooth_window: int | None = None,
    unwrap: bool = True,
) -> pd.DataFrame:
    """Extract the cycle phase θ(t) from the non-linear ring in BTC features.

    Returns a DataFrame with columns:
        theta       — phase angle in [-π, π] (or unwrapped if unwrap=True)
        cos_theta   — cosine of phase (use as circular feature for ML)
        sin_theta   — sine of phase   (use as circular feature for ML)
        umap_x      — UMAP-2D x coordinate
        umap_y      — UMAP-2D y coordinate
        radius      — distance from ring origin (confidence proxy: near 1 = on ring)

    Args:
        df:              DataFrame with OHLCV + microstructure columns.
        feature_mode:    Which feature group to use (default 'compact').
        normalize_method: Normalisation applied before UMAP.
        n_neighbors:     UMAP neighbourhood size.
        min_dist:        UMAP min_dist.
        seed:            Reproducibility seed.
        timestamp_col:   Column to use as index; None to use integer index.
        smooth_window:   If set, apply a rolling circular mean over this window.
        unwrap:          Unwrap phase to remove 2π jumps (gives continuous signal).

    Returns:
        pd.DataFrame aligned to the valid (non-NaN) rows of df.
    """
    feat = build_financial_features(df)
    groups = feature_column_groups()
    cols = [c for c in groups.get(feature_mode, groups["compact"]) if c in feat.columns]
    feat_clean = feat[cols].dropna()

    data = normalize(feat_clean.to_numpy(dtype=float), method=normalize_method)
    embedding = _embed_umap_2d(data, n_neighbors=n_neighbors,
                                min_dist=min_dist, seed=seed)

    x, y = embedding[:, 0], embedding[:, 1]
    theta = np.arctan2(y, x)
    radius = np.sqrt(x ** 2 + y ** 2)

    if smooth_window and smooth_window > 1:
        # Circular mean: average real and imaginary parts separately
        z = np.exp(1j * theta)
        w = int(smooth_window)
        z_re = pd.Series(z.real).rolling(w, center=True, min_periods=1).mean().to_numpy()
        z_im = pd.Series(z.imag).rolling(w, center=True, min_periods=1).mean().to_numpy()
        theta = np.angle(z_re + 1j * z_im)

    if unwrap:
        theta = np.unwrap(theta)

    out = pd.DataFrame({
        "theta":     theta,
        "cos_theta": np.cos(theta),
        "sin_theta": np.sin(theta),
        "umap_x":    x,
        "umap_y":    y,
        "radius":    radius,
    }, index=feat_clean.index)

    if timestamp_col and timestamp_col in df.columns:
        out.insert(0, timestamp_col, df[timestamp_col].iloc[feat_clean.index].values)

    return out


def analyse_cycle_period(
    phase: pd.DataFrame | pd.Series,
    bars_per_day: float = 24.0,
    top_n: int = 5,
) -> dict:
    """Estimate the dominant period(s) of the cycle phase via FFT on dθ/dt.

    Args:
        phase:          Output of extract_cycle_phase() or a plain Series of θ.
        bars_per_day:   Number of bars per calendar day (24 for 1h, 480 for 3m).
        top_n:          Number of top frequency peaks to return.

    Returns:
        dict with keys:
            periods_bars    — list of dominant periods in bars
            periods_days    — same in calendar days
            periods_named   — human labels (e.g. "~weekly", "~monthly")
            angular_velocity_mean — mean |dθ/dt| (rad/bar)
            estimated_period_bars — 2π / mean|dθ/dt| (naive estimate)
            fft_freqs, fft_power  — raw FFT spectrum for plotting
    """
    if isinstance(phase, pd.DataFrame):
        theta = phase["theta"].to_numpy(dtype=float)
    else:
        theta = np.asarray(phase, dtype=float)

    # Angular velocity from unwrapped phase
    dtheta = np.diff(theta)
    omega_mean = float(np.abs(dtheta).mean())
    naive_period_bars = float(2 * np.pi / omega_mean) if omega_mean > 0 else float("nan")
    naive_period_days = naive_period_bars / bars_per_day

    # FFT on angular velocity to find dominant frequencies
    n = len(dtheta)
    fft_vals = np.fft.rfft(dtheta - dtheta.mean())
    fft_power = np.abs(fft_vals) ** 2
    fft_freqs = np.fft.rfftfreq(n)  # cycles per bar

    # Find top peaks (excluding DC)
    peak_idx = np.argsort(fft_power[1:])[::-1][:top_n] + 1
    periods_bars = [float(1.0 / fft_freqs[i]) if fft_freqs[i] > 0 else float("nan")
                    for i in peak_idx]
    periods_days = [p / bars_per_day for p in periods_bars]

    def _label(days: float) -> str:
        if not np.isfinite(days):
            return "unknown"
        if days < 1.2:
            return f"~intraday ({days*24:.0f}h)"
        if days < 2.5:
            return f"~2day ({days:.1f}d)"
        if 5 < days < 9:
            return f"~weekly ({days:.1f}d)"
        if 12 < days < 18:
            return f"~biweekly ({days:.1f}d)"
        if 25 < days < 35:
            return f"~monthly ({days:.1f}d)"
        if 80 < days < 100:
            return f"~quarterly ({days:.0f}d)"
        if 350 < days < 380:
            return f"~annual ({days:.0f}d)"
        if 1300 < days < 1600:
            return f"~4yr-halving ({days:.0f}d)"
        return f"{days:.1f}d"

    periods_named = [_label(d) for d in periods_days]

    return {
        "angular_velocity_mean": omega_mean,
        "estimated_period_bars": naive_period_bars,
        "estimated_period_days": naive_period_days,
        "estimated_period_named": _label(naive_period_days),
        "top_periods_bars": periods_bars,
        "top_periods_days": periods_days,
        "top_periods_named": periods_named,
        "fft_freqs": fft_freqs.tolist(),
        "fft_power": fft_power.tolist(),
    }


def ring_quality(phase: pd.DataFrame) -> dict:
    """Quantify how ring-like the UMAP-2D embedding is.

    Returns:
        radius_cv   — coefficient of variation of radius (< 0.3 = ring)
        angle_uniformity — std of sorted angle differences (< 0.01 = uniform)
        ring_score  — composite: 1 - radius_cv (higher = more ring-like)
    """
    r = phase["radius"].to_numpy(dtype=float)
    theta_raw = np.arctan2(phase["umap_y"].to_numpy(), phase["umap_x"].to_numpy())
    radius_cv = float(np.std(r) / np.mean(r))
    sorted_angles = np.sort(theta_raw % (2 * np.pi))
    angle_diffs = np.diff(sorted_angles)
    angle_uniformity = float(np.std(angle_diffs))
    ring_score = float(max(0.0, 1.0 - radius_cv))
    return {
        "radius_cv": radius_cv,
        "angle_uniformity": angle_uniformity,
        "ring_score": ring_score,
        "is_ring": radius_cv < 0.3,
    }
