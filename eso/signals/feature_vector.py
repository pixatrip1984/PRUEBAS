"""Final feature vector composition for downstream ML models.

Combines causal cycle phase (geometric) with financial context features
to produce a complete, ML-ready feature matrix.

Validated feature set (r_OOS=0.320, BTCUSDT 1h, 2024-2026):
    sin_theta_6h   — short-cycle position (12h period)
    cos_theta_6h   — orthogonal short-cycle
    sin_theta_24h  — medium-cycle position (biweekly ~11d)
    cos_theta_24h  — orthogonal medium-cycle
    sin_theta_72h  — longer-cycle position (~monthly)
    cos_theta_72h  — orthogonal longer-cycle
    ring_radius    — UMAP ring confidence (larger = more on-ring)
    vol_20         — rolling 20-bar volatility (amplitude context)
    lr_z20         — return z-score over 20 bars (momentum context)

Usage:
    from eso.signals.feature_vector import build_feature_vector, CausalFeatureBuilder

    # One-shot (transductive, for research)
    df = pd.read_csv('data/BTCUSDT_1h.csv')
    fv = build_feature_vector(df, train_size=27000)

    # Production (causal rolling)
    builder = CausalFeatureBuilder(init_train=10000, refit_every=2000)
    fv = builder.fit_predict(df)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from eso.data.features import build_financial_features, feature_column_groups
from eso.data.preprocess import normalize
from eso.signals.causal_cycle import CausalCyclePhase, RollingCausalCycle


# Canonical feature column order for downstream models
FEATURE_COLUMNS = [
    "sin_theta_6h",  "cos_theta_6h",
    "sin_theta_24h", "cos_theta_24h",
    "sin_theta_72h", "cos_theta_72h",
    "ring_radius",
    "vol_20",
    "lr_z20",
]


def _smooth_phase(theta: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) of phase smoothed over `window` bars."""
    z_re = pd.Series(np.cos(theta)).rolling(window, center=False, min_periods=max(1, window//4)).mean().to_numpy()
    z_im = pd.Series(np.sin(theta)).rolling(window, center=False, min_periods=max(1, window//4)).mean().to_numpy()
    theta_s = np.angle(z_re + 1j * z_im)
    return np.sin(theta_s), np.cos(theta_s)


def _add_cycle_features(
    out: pd.DataFrame,
    theta: np.ndarray,
    radius: np.ndarray,
    smooth_windows: tuple[int, ...] = (6, 24, 72),
) -> pd.DataFrame:
    for win in smooth_windows:
        s, c = _smooth_phase(theta, win)
        out[f"sin_theta_{win}h"] = s
        out[f"cos_theta_{win}h"] = c
    out["ring_radius"] = radius
    return out


def build_feature_vector(
    df: pd.DataFrame,
    train_size: int = 27000,
    smooth_windows: tuple[int, ...] = (6, 24, 72),
    seed: int = 42,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Build the full feature vector using a causal train/test split.

    The geometric features (cycle phase) are derived causally: UMAP is fitted
    only on the first `train_size` bars and used to project the rest.
    The financial context features (vol_20, lr_z20) are computed without
    future information by construction (rolling windows looking backward).

    Args:
        df:            Raw OHLCV + microstructure DataFrame.
        train_size:    Number of bars for UMAP training window.
        smooth_windows: Phase smoothing windows in bars.
        seed:          UMAP reproducibility seed.
        drop_warmup:   Drop early rows where rolling features are NaN-filled.

    Returns:
        DataFrame with columns in FEATURE_COLUMNS order plus 'timestamp',
        'log_return' (target proxy), and 'close' (for label construction).
    """
    # Build financial features (all causal by construction)
    feat = build_financial_features(df)
    groups = feature_column_groups()
    compact_cols = [c for c in groups["compact"] if c in feat.columns]
    feat_clean = feat.dropna(subset=compact_cols)
    orig_idx = feat_clean.index.to_numpy()

    n = len(feat_clean)
    train_n = min(train_size, int(n * 0.6))

    # Fit UMAP on training window only
    causal = CausalCyclePhase(train_size=train_n, seed=seed)
    feat_train = feat_clean[compact_cols].iloc[:train_n]
    feat_test  = feat_clean[compact_cols].iloc[train_n:]

    causal.fit(feat_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_test = causal.transform(feat_test)

    theta = np.unwrap(np.arctan2(emb_test[:, 1], emb_test[:, 0]))
    radius = np.sqrt(emb_test[:, 0]**2 + emb_test[:, 1]**2)

    # Assemble output aligned to the OOS test window
    test_idx = orig_idx[train_n:]
    out = pd.DataFrame(index=feat_test.index)

    # Geometric features
    out = _add_cycle_features(out, theta, radius, smooth_windows)

    # Financial context features (already causal — feat_clean is indexed by df positions)
    for col in ["vol_20", "lr_z20", "log_return"]:
        if col in feat_clean.columns:
            out[col] = feat_clean[col].iloc[train_n:].values

    # Labels and metadata
    out["close"] = df["close"].to_numpy(dtype=float)[test_idx]

    if "timestamp" in df.columns:
        out.insert(0, "timestamp", df["timestamp"].iloc[test_idx].values)

    if drop_warmup:
        # Drop first max(smooth_windows) rows to avoid NaN-filled rolling features
        warmup = max(smooth_windows)
        out = out.iloc[warmup:].copy()

    return out


class CausalFeatureBuilder:
    """Production-ready feature vector builder using rolling UMAP.

    Usage:
        builder = CausalFeatureBuilder(init_train=10000, refit_every=2000)
        fv = builder.fit_predict(df)
        # fv has columns: sin_theta_6h, cos_theta_6h, ..., vol_20, lr_z20, close

    Args:
        init_train:     Initial UMAP training window in bars.
        refit_every:    Refit UMAP every N new bars.
        smooth_windows: Phase smoothing windows for feature extraction.
        seed:           Reproducibility seed.
    """

    def __init__(
        self,
        init_train: int = 10000,
        refit_every: int = 2000,
        smooth_windows: tuple[int, ...] = (6, 24, 72),
        seed: int = 42,
    ) -> None:
        self.init_train = init_train
        self.refit_every = refit_every
        self.smooth_windows = smooth_windows
        self.seed = seed
        self._rolling = RollingCausalCycle(
            init_train=init_train,
            refit_every=refit_every,
            seed=seed,
        )

    def fit_predict(self, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """Run rolling UMAP and return the complete feature vector.

        All outputs are causal: feature at time t uses only data up to t.
        """
        # Get rolling phase (already causal)
        phase = self._rolling.fit_predict(df, timestamp_col="timestamp")

        if phase.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        theta = phase["theta"].to_numpy(dtype=float)
        radius = phase["radius"].to_numpy(dtype=float)

        out = pd.DataFrame(index=phase.index)
        out = _add_cycle_features(out, theta, radius, self.smooth_windows)

        # Financial context — select from feat aligned to phase.index
        feat = build_financial_features(df)
        for col in ["vol_20", "lr_z20"]:
            if col in feat.columns:
                out[col] = feat[col].iloc[phase.index].values

        # Metadata
        orig_idx = phase.index.to_numpy()
        out["close"] = df["close"].to_numpy(dtype=float)[orig_idx]

        if "log_return" in feat.columns:
            # phase.index contains positional indices into feat_clean
            feat_clean = feat.dropna(subset=["log_return"])
            # phase.index = positions in feat_clean; use direct positional lookup
            out["log_return"] = feat_clean["log_return"].iloc[phase.index.to_numpy()].values

        if "timestamp" in phase.columns:
            out.insert(0, "timestamp", phase["timestamp"].values)

        if drop_warmup:
            warmup = max(self.smooth_windows)
            out = out.iloc[warmup:].copy()

        return out

    @property
    def feature_columns(self) -> list[str]:
        cols = []
        for w in self.smooth_windows:
            cols += [f"sin_theta_{w}h", f"cos_theta_{w}h"]
        cols += ["ring_radius", "vol_20", "lr_z20"]
        return cols
