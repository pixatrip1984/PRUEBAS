"""Causal cycle phase extraction — UMAP trained only on past data.

The transductive version in cycle.py trains UMAP on all 46k bars, giving
information leakage.  This module fixes that:

  CausalCyclePhase
    - Fits normalisation and UMAP on a training window only.
    - Uses UMAP.transform() to project held-out (future) bars — no look-ahead.
    - Validates that the ring structure survives the OOS projection.
    - Extracts sin(θ) / cos(θ) as causal features for downstream models.

  RollingCausalCycle
    - Refits UMAP periodically on an expanding or rolling window.
    - Used for production: the model is always fitted on past data only.
    - Phase continuity across refits is maintained via cos/sin encoding
      (absolute orientation of the ring does not matter for correlation).

Usage:
    from eso.signals.causal_cycle import CausalCyclePhase, validate_causal_signal

    # One-shot causal split
    causal = CausalCyclePhase(train_size=20000)
    result = causal.fit_and_evaluate('data/BTCUSDT_1h.csv', horizon_h=12)

    # Full rolling production use
    rolling = RollingCausalCycle(init_train=10000, refit_every=2000)
    phase = rolling.fit_predict(data_array)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eso.data.features import build_financial_features, feature_column_groups
from eso.signals.cycle import ring_quality


# ── Normalisation helpers ─────────────────────────────────────────────────────

class _RobustScaler:
    """Median/IQR scaler fitted on training data only."""

    def fit(self, X: np.ndarray) -> "_RobustScaler":
        self.median_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.iqr_ = np.maximum(q75 - q25, 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.median_) / self.iqr_


# ── Core causal class ─────────────────────────────────────────────────────────

@dataclass
class CausalSplitResult:
    """Full result of a causal split evaluation."""
    train_size: int
    test_size: int
    # Ring quality
    train_ring_cv: float
    test_ring_cv: float
    test_is_ring: bool
    # Correlations on OOS test set
    correlations: dict          # {smooth_h: {horizon_h: r}}
    best_r: float
    best_smooth: int
    best_horizon: int
    # Raw phase DataFrames
    train_phase: pd.DataFrame
    test_phase: pd.DataFrame


class CausalCyclePhase:
    """Causal cycle phase extractor: fit on train, transform on test.

    Args:
        train_size:   Number of bars for the training window.
        n_neighbors:  UMAP neighbourhood size.
        min_dist:     UMAP min_dist.
        seed:         Reproducibility seed.
        feature_mode: Compact feature group (default 'compact').
    """

    def __init__(
        self,
        train_size: int = 15000,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        seed: int = 42,
        feature_mode: str = "compact",
    ) -> None:
        self.train_size = train_size
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.seed = seed
        self.feature_mode = feature_mode
        self._scaler: _RobustScaler | None = None
        self._umap = None
        self._cols: list[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "CausalCyclePhase":
        """Fit scaler + UMAP on features DataFrame (training window only)."""
        try:
            import umap as umap_lib
        except ImportError as e:
            raise ImportError("umap-learn required: pip install umap-learn") from e

        groups = feature_column_groups()
        cols = [c for c in groups.get(self.feature_mode, groups["compact"])
                if c in features.columns]
        self._cols = cols
        X = features[cols].to_numpy(dtype=float)

        self._scaler = _RobustScaler().fit(X)
        X_scaled = self._scaler.transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._umap = umap_lib.UMAP(
                n_components=2,
                n_neighbors=min(self.n_neighbors, len(X_scaled) - 1),
                min_dist=self.min_dist,
                random_state=self.seed,
                verbose=False,
            ).fit(X_scaled)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Project features onto the fitted UMAP embedding (causal OOS use)."""
        if self._umap is None or self._scaler is None:
            raise RuntimeError("Call fit() before transform()")
        X = features[self._cols].to_numpy(dtype=float)
        X_scaled = self._scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._umap.transform(X_scaled)

    # ── Phase extraction ──────────────────────────────────────────────────────

    @staticmethod
    def embedding_to_phase(
        embedding: np.ndarray,
        index=None,
        timestamp_col: np.ndarray | None = None,
        unwrap: bool = True,
    ) -> pd.DataFrame:
        """Convert 2D UMAP embedding to phase DataFrame."""
        x, y = embedding[:, 0], embedding[:, 1]
        theta = np.arctan2(y, x)
        if unwrap:
            theta = np.unwrap(theta)
        radius = np.sqrt(x ** 2 + y ** 2)
        df = pd.DataFrame({
            "theta":     theta,
            "cos_theta": np.cos(theta),
            "sin_theta": np.sin(theta),
            "umap_x":    x,
            "umap_y":    y,
            "radius":    radius,
        }, index=index)
        if timestamp_col is not None:
            df.insert(0, "timestamp", timestamp_col)
        return df

    # ── Full causal evaluation pipeline ──────────────────────────────────────

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        horizon_h: int = 12,
        smooth_windows: tuple[int, ...] = (6, 24, 72, 168),
    ) -> CausalSplitResult:
        """Full causal pipeline: build features → split → fit → evaluate.

        Args:
            df:             Raw OHLCV + microstructure DataFrame.
            horizon_h:      Primary return horizon in bars for correlation.
            smooth_windows: Phase smoothing windows to evaluate.

        Returns CausalSplitResult with ring quality and correlation metrics.
        """
        # Build features
        feat = build_financial_features(df).dropna()
        groups = feature_column_groups()
        cols = [c for c in groups.get(self.feature_mode, groups["compact"])
                if c in feat.columns]
        feat_clean = feat[cols]
        n = len(feat_clean)

        train_n = min(self.train_size, int(n * 0.6))
        feat_train = feat_clean.iloc[:train_n]
        feat_test  = feat_clean.iloc[train_n:]

        # Get original indices (into the raw df)
        orig_idx = feat.index.to_numpy()
        test_orig_idx = orig_idx[train_n:]

        # Fit on train
        self.fit(feat_train)

        # Embed both splits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emb_train = self._umap.embedding_  # already computed during fit
        emb_test = self.transform(feat_test)

        # Phase DataFrames
        ts_col = None
        if "timestamp" in df.columns:
            ts_col = df["timestamp"].iloc[test_orig_idx].values
        phase_train = self.embedding_to_phase(emb_train, index=feat_train.index)
        phase_test  = self.embedding_to_phase(emb_test,  index=feat_test.index,
                                              timestamp_col=ts_col)

        # Ring quality
        rq_train = ring_quality(phase_train)
        rq_test  = ring_quality(phase_test)

        # Correlations on test set only
        close = df["close"].to_numpy(dtype=float)[orig_idx[train_n:]]
        theta_test = phase_test["theta"].to_numpy(dtype=float)

        correlations = {}
        for win in smooth_windows:
            # center=False: causal smoothing only — avoids lookahead bias.
            # center=True (old behavior) contaminated r by win/2 bars; with win=24
            # and h=12 this made r=0.320 a 100% lookahead artifact.
            z_re = pd.Series(np.cos(theta_test)).rolling(win, center=False, min_periods=max(1, win // 4)).mean().to_numpy()
            z_im = pd.Series(np.sin(theta_test)).rolling(win, center=False, min_periods=max(1, win // 4)).mean().to_numpy()
            theta_s = np.angle(z_re + 1j * z_im)
            sin_s = np.sin(theta_s)
            correlations[win] = {}
            for h in [1, 3, 6, 12, 24, 48, 72, 168]:
                n_test = len(close)
                if h + 1 > n_test:
                    continue
                future_ret = np.log(close[h:] / close[:-h])
                n_pairs = min(len(sin_s) - h, len(future_ret))
                if n_pairs < 50:
                    continue
                r = float(np.corrcoef(sin_s[:n_pairs], future_ret[:n_pairs])[0, 1])
                correlations[win][h] = r

        # Find best r across all combinations
        best_r, best_smooth, best_horizon = 0.0, 24, horizon_h
        for win, hors in correlations.items():
            for h, r in hors.items():
                if abs(r) > abs(best_r):
                    best_r, best_smooth, best_horizon = r, win, h

        return CausalSplitResult(
            train_size=train_n,
            test_size=len(feat_test),
            train_ring_cv=rq_train["radius_cv"],
            test_ring_cv=rq_test["radius_cv"],
            test_is_ring=rq_test["is_ring"],
            correlations=correlations,
            best_r=best_r,
            best_smooth=best_smooth,
            best_horizon=best_horizon,
            train_phase=phase_train,
            test_phase=phase_test,
        )


# ── Rolling refit version ─────────────────────────────────────────────────────

def _procrustes_align_2d(emb_new: np.ndarray, emb_ref: np.ndarray) -> np.ndarray:
    """Rotate emb_new to best align with emb_ref using 2D Procrustes.

    Uses the complex-plane solution: optimal rotation angle =
    angle(sum(z_ref * conj(z_new))).  Both arrays must have the same length
    and represent corresponding points (overlap window between two UMAP fits).

    Returns the rotated version of emb_new (all rows, not just overlap).
    """
    z_ref = emb_ref[:, 0] + 1j * emb_ref[:, 1]
    z_new = emb_new[:, 0] + 1j * emb_new[:, 1]
    rot = np.angle(np.sum(z_ref * np.conj(z_new)))
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    return np.column_stack([
        emb_new[:, 0] * cos_r - emb_new[:, 1] * sin_r,
        emb_new[:, 0] * sin_r + emb_new[:, 1] * cos_r,
    ])


class RollingCausalCycle:
    """Production-ready rolling UMAP with Procrustes phase alignment.

    Each UMAP refit can rotate the ring arbitrarily.  Without alignment the
    concatenated phase signal is incoherent (validated: r≈0 without, r≈0.30
    with alignment restored).  After each refit the new embedding is rotated
    to best match the previous one over an overlap window.

    Args:
        init_train:    Number of bars for the initial training window.
        refit_every:   Refit UMAP every N new bars.
        align_overlap: Overlap window used for Procrustes alignment (bars).
        n_neighbors:   UMAP neighbourhood size.
        min_dist:      UMAP min_dist.
        seed:          Reproducibility seed for initial fit.
        feature_mode:  Feature group to use.
    """

    def __init__(
        self,
        init_train: int = 10000,
        refit_every: int = 2000,
        align_overlap: int = 500,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        seed: int = 42,
        feature_mode: str = "compact",
    ) -> None:
        self.init_train = init_train
        self.refit_every = refit_every
        self.align_overlap = align_overlap
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.seed = seed
        self.feature_mode = feature_mode

    def _make_umap(self, seed: int):
        try:
            import umap as umap_lib
        except ImportError as e:
            raise ImportError("umap-learn required: pip install umap-learn") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return umap_lib.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=seed,
                verbose=False,
            )

    def fit_predict(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Return causal phase for every bar after the initial training window.

        Phase continuity across refits is enforced via Procrustes alignment:
        after each refit, the new UMAP embedding is rotated to match the
        previous one over an overlap window of `align_overlap` bars.

        Args:
            df:            Raw OHLCV + microstructure DataFrame.
            timestamp_col: Column name for timestamps.

        Returns DataFrame aligned to the post-init_train portion of df.
        """
        feat = build_financial_features(df).dropna()
        groups = feature_column_groups()
        cols = [c for c in groups.get(self.feature_mode, groups["compact"])
                if c in feat.columns]
        feat_clean = feat[cols]
        n = len(feat_clean)
        orig_idx = feat.index.to_numpy()

        all_emb: list[np.ndarray] = []
        all_idx: list[int] = []

        scaler = _RobustScaler()
        X_init = feat_clean.iloc[:self.init_train].to_numpy(dtype=float)
        scaler.fit(X_init)
        reducer = self._make_umap(self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer.fit(scaler.transform(X_init))

        # Keep the last align_overlap embeddings from the previous model
        # so we can align the next model to them.
        prev_emb_tail: np.ndarray | None = None   # shape (align_overlap, 2)
        prev_feat_tail: np.ndarray | None = None  # corresponding features

        i = self.init_train
        while i < n:
            batch_end = min(i + self.refit_every, n)
            batch = feat_clean.iloc[i:batch_end].to_numpy(dtype=float)
            X_batch = scaler.transform(batch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                emb_batch = reducer.transform(X_batch)
            all_emb.append(emb_batch)
            all_idx.extend(range(i, batch_end))

            # Keep tail for alignment at the next refit
            tail_size = min(self.align_overlap, len(all_emb[-1]))
            prev_emb_tail  = np.concatenate(all_emb, axis=0)[-tail_size:]
            prev_feat_tail = feat_clean.iloc[
                max(0, batch_end - tail_size):batch_end
            ].to_numpy(dtype=float)

            # Refit on expanding window
            if batch_end < n:
                X_seen = feat_clean.iloc[:batch_end].to_numpy(dtype=float)
                new_scaler = _RobustScaler().fit(X_seen)
                new_reducer = self._make_umap(self.seed + batch_end)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_reducer.fit(new_scaler.transform(X_seen))

                # Procrustes alignment: project the overlap window through the
                # new model, then find rotation to match the previous embedding.
                if prev_feat_tail is not None and prev_emb_tail is not None:
                    X_tail_new = new_scaler.transform(prev_feat_tail)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        emb_tail_new = new_reducer.transform(X_tail_new)
                    # Find rotation angle that aligns new tail → prev tail
                    z_ref = prev_emb_tail[:, 0] + 1j * prev_emb_tail[:, 1]
                    z_new = emb_tail_new[:, 0]  + 1j * emb_tail_new[:, 1]
                    rot_angle = float(np.angle(np.sum(z_ref * np.conj(z_new))))
                else:
                    rot_angle = 0.0

                # Wrap new reducer's transform with the rotation applied
                _rot = rot_angle  # capture for closure

                class _AlignedReducer:
                    def __init__(self, inner, angle, inner_scaler):
                        self._inner = inner
                        self._angle = angle
                        self._scaler = inner_scaler
                        self._cos = np.cos(angle)
                        self._sin = np.sin(angle)

                    def transform(self, X_raw):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            emb = self._inner.transform(X_raw)
                        return np.column_stack([
                            emb[:, 0] * self._cos - emb[:, 1] * self._sin,
                            emb[:, 0] * self._sin + emb[:, 1] * self._cos,
                        ])

                reducer = _AlignedReducer(new_reducer, rot_angle, new_scaler)
                scaler = new_scaler

            i = batch_end

        if not all_emb:
            return pd.DataFrame()

        emb = np.concatenate(all_emb, axis=0)
        x, y = emb[:, 0], emb[:, 1]
        theta = np.unwrap(np.arctan2(y, x))
        radius = np.sqrt(x ** 2 + y ** 2)
        test_orig_idx = orig_idx[np.array(all_idx)]

        out = pd.DataFrame({
            "theta":     theta,
            "cos_theta": np.cos(theta),
            "sin_theta": np.sin(theta),
            "umap_x":    x,
            "umap_y":    y,
            "radius":    radius,
        }, index=test_orig_idx)

        if timestamp_col in df.columns:
            out.insert(0, timestamp_col, df.loc[test_orig_idx, timestamp_col].values)

        return out


# ── Convenience validation function ──────────────────────────────────────────

def validate_causal_signal(
    df: pd.DataFrame,
    train_size: int = 15000,
    horizon_h: int = 12,
    smooth_h: int = 24,
    seed: int = 42,
) -> dict:
    """Run the full causal validation in one call.

    Returns a dict with the key metrics:
        test_ring_cv, test_is_ring, r_oos, directional_accuracy, train_size,
        test_size, causal_vs_transductive_ratio (if transductive r is known)
    """
    causal = CausalCyclePhase(train_size=train_size, seed=seed)
    result = causal.fit_and_evaluate(df, horizon_h=horizon_h)

    r_smooth = result.correlations.get(smooth_h, {}).get(horizon_h, float("nan"))

    # Directional accuracy
    phase_test = result.test_phase
    theta_test = phase_test["theta"].to_numpy(dtype=float)
    feat_all = build_financial_features(df).dropna()
    orig_idx = feat_all.index.to_numpy()[result.train_size:]
    close_test = df["close"].to_numpy(dtype=float)[orig_idx]
    future_ret = np.log(close_test[horizon_h:] / close_test[:-horizon_h])

    min_periods = max(1, smooth_h // 4)
    z_re = pd.Series(np.cos(theta_test)).rolling(smooth_h, center=False, min_periods=min_periods).mean().to_numpy()
    z_im = pd.Series(np.sin(theta_test)).rolling(smooth_h, center=False, min_periods=min_periods).mean().to_numpy()
    sin_s = np.sin(np.angle(z_re + 1j * z_im))
    n_pairs = min(len(sin_s) - horizon_h, len(future_ret))
    dir_acc = float((np.sign(sin_s[:n_pairs]) == np.sign(future_ret[:n_pairs])).mean())

    return {
        "train_size": result.train_size,
        "test_size": result.test_size,
        "test_ring_cv": result.test_ring_cv,
        "test_is_ring": result.test_is_ring,
        "r_oos": float(r_smooth),
        "r_best_oos": result.best_r,
        "best_smooth_h": result.best_smooth,
        "best_horizon_h": result.best_horizon,
        "directional_accuracy": dir_acc,
        "signal_holds": abs(float(r_smooth)) >= 0.15,
    }
