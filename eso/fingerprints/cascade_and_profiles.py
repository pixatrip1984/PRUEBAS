"""Causal cascade dynamics and capital/execution regime segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

_EPS = 1e-12


def _num(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _returns(df: pd.DataFrame) -> pd.Series:
    if "log_return" in df.columns:
        return _num(df, "log_return")
    close = _num(df, "close").where(lambda x: x > 0.0)
    return np.log(close).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _causal_z(s: pd.Series, window: int) -> pd.Series:
    minp = max(2, min(window, max(2, window // 3)))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    return ((s - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class CascadeAnalyzer:
    """Analyze shock clusters as causal avalanche/self-excitation processes."""

    window: int = 40
    shock_z: float = 2.0
    max_gap: int = 2
    decay: float = 0.7

    def _events(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        ret = _returns(df)
        vol = _num(df, "volume", 0.0).clip(lower=0.0)
        rz = _causal_z(ret.abs(), self.window)
        vz = _causal_z(np.log1p(vol), self.window)
        intensity = np.maximum(
            rz.clip(lower=0.0),
            0.5 * rz.clip(lower=0.0) + 0.5 * vz.clip(lower=0.0),
        )
        shock = (intensity >= self.shock_z).astype(int)
        return (
            shock,
            pd.Series(intensity, index=df.index)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0),
        )

    def _avalanche_ids(self, shock: pd.Series) -> np.ndarray:
        ids = np.full(len(shock), -1, dtype=int)
        event_pos = np.flatnonzero(shock.to_numpy(dtype=bool))
        if not len(event_pos):
            return ids
        aid = 0
        prev = event_pos[0]
        ids[prev] = aid
        for pos in event_pos[1:]:
            if pos - prev > self.max_gap + 1:
                aid += 1
            ids[pos] = aid
            prev = pos
        return ids

    def analyze(self, df: pd.DataFrame) -> dict[str, object]:
        if self.window < 3 or self.max_gap < 0:
            raise ValueError("window must be >= 3 and max_gap >= 0")
        shock, intensity = self._events(df)
        ids = self._avalanche_ids(shock)
        event_pos = np.flatnonzero(shock.to_numpy(dtype=bool))

        avalanches: list[dict[str, float | int]] = []
        if len(event_pos):
            for aid in range(int(ids.max()) + 1):
                pos = np.flatnonzero(ids == aid)
                if not len(pos):
                    continue
                start, end = int(pos[0]), int(pos[-1])
                size = int(len(pos))
                duration = int(end - start + 1)
                local = intensity.iloc[start : end + 1]
                peak = float(local.max()) if len(local) else 0.0
                seed = float(max(intensity.iloc[start], _EPS))
                avalanches.append(
                    {
                        "start": start,
                        "end": end,
                        "size": size,
                        "duration": duration,
                        "peak_amplification": float(peak / seed),
                    }
                )

        triggered = sum(max(int(a["size"]) - 1, 0) for a in avalanches)
        total_events = int(shock.sum())
        exogenous = len(avalanches)
        eta = float(triggered / exogenous) if exogenous else 0.0

        memory = np.zeros(len(shock), dtype=float)
        state = 0.0
        for i, flag in enumerate(shock.to_numpy(dtype=float)):
            state = self.decay * state + flag
            memory[i] = state
        memory_s = pd.Series(memory, index=df.index)
        causal_peak = (
            memory_s.rolling(self.window, min_periods=1).max().clip(lower=1.0)
        )
        risk = (memory_s / causal_peak).clip(0.0, 1.0)

        amp_parts = []
        for scale in (2, 4, 8):
            rolling_count = shock.rolling(scale, min_periods=1).sum()
            baseline = shock.expanding(min_periods=1).mean() * scale
            amp_parts.append(
                (rolling_count / baseline.replace(0.0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        multiscale = pd.concat(amp_parts, axis=1).mean(axis=1)

        return {
            "branching_ratio_eta": eta,
            "event_count": total_events,
            "avalanche_count": len(avalanches),
            "avalanches": avalanches,
            "mean_avalanche_size": float(
                np.mean([a["size"] for a in avalanches])
            )
            if avalanches
            else 0.0,
            "mean_avalanche_duration": float(
                np.mean([a["duration"] for a in avalanches])
            )
            if avalanches
            else 0.0,
            "max_peak_amplification": float(
                max((a["peak_amplification"] for a in avalanches), default=0.0)
            ),
            "rolling_cascade_risk": risk,
            "multiscale_snowball_amplification": multiscale.replace(
                [np.inf, -np.inf], 0.0
            ).fillna(0.0),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.analyze(df)
        return pd.DataFrame(
            {
                "cascade_risk_score": result["rolling_cascade_risk"],
                "snowball_amplification": result[
                    "multiscale_snowball_amplification"
                ],
            },
            index=df.index,
        )


@dataclass
class CapitalProfileSegmenter:
    """Classify causal execution regimes from ticket size, VWAP discipline and impact."""

    window: int = 30
    min_periods: int = 5
    vwap_tolerance: float = 0.002
    volatility_threshold: float = 0.002

    labels: tuple[str, str, str] = (
        "fragmented_retail",
        "patient_institutional",
        "aggressive_momentum",
    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.window < 3:
            raise ValueError("window must be >= 3")
        ret = _returns(df)
        volume = _num(df, "volume", 0.0).clip(lower=0.0)
        close = _num(df, "close", 0.0).abs()
        trades = _num(df, "n_trades", 0.0).clip(lower=0.0)

        if "n_trades" in df.columns:
            ticket = volume / trades.replace(0.0, np.nan)
        else:
            ticket = volume
        ticket = ticket.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if "vwap" in df.columns:
            vwap = _num(df, "vwap", 0.0)
            vwap_dev = (close - vwap).abs() / close.replace(0.0, np.nan)
        else:
            vwap_dev = ret.abs()
        vwap_dev = (
            vwap_dev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        vol = ret.rolling(self.window, min_periods=1).std(ddof=0).fillna(0.0)
        impact = ret.abs() / (close * volume).replace(0.0, np.nan)
        impact = impact.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        def causal_rank(s: pd.Series) -> pd.Series:
            return (
                s.rolling(self.window, min_periods=1)
                .apply(
                    lambda x: float(np.mean(x <= x[-1]))
                    if len(x)
                    else 0.5,
                    raw=True,
                )
                .fillna(0.5)
            )

        ticket_rank = causal_rank(ticket)

        if "n_trades" in df.columns:
            trade_density = trades / (trades + volume).replace(0.0, np.nan)
            trade_density = trade_density.fillna(0.0).clip(0.0, 1.0)
            fragmentation = (
                trade_density / (trade_density + 0.10)
            ).fillna(0.0).clip(0.0, 1.0)
        else:
            fragmentation = (1.0 - ticket_rank).clip(0.0, 1.0)

        vwap_scale = max(float(self.vwap_tolerance), _EPS)
        vol_scale = max(float(self.volatility_threshold), _EPS)
        dislocation = (
            vwap_dev / (vwap_dev + vwap_scale)
        ).fillna(0.0).clip(0.0, 1.0)
        discipline = 1.0 - dislocation
        volatility_load = (
            vol / (vol + vol_scale)
        ).fillna(0.0).clip(0.0, 1.0)

        retail = (
            0.70 * fragmentation
            + 0.15 * (1.0 - ticket_rank)
            + 0.15 * volatility_load
        )
        institutional = (
            0.35 * (1.0 - fragmentation)
            + 0.30 * discipline
            + 0.20 * (1.0 - volatility_load)
            + 0.15 * ticket_rank
        )
        aggressive = (
            0.30 * (1.0 - fragmentation)
            + 0.30 * dislocation
            + 0.25 * volatility_load
            + 0.15 * ticket_rank
        )

        scores = pd.DataFrame(
            {
                self.labels[0]: retail,
                self.labels[1]: institutional,
                self.labels[2]: aggressive,
            },
            index=df.index,
        ).clip(lower=0.0)
        denom = scores.sum(axis=1).replace(0.0, np.nan)
        probs = scores.div(denom, axis=0).fillna(1.0 / 3.0)
        regime = probs.idxmax(axis=1)
        confidence = probs.max(axis=1)

        return pd.DataFrame(
            {
                "capital_regime": regime,
                "confidence": confidence,
                "ticket_size_proxy": ticket,
                "vwap_deviation": vwap_dev,
                "realized_volatility": vol,
                "impact_proxy": impact,
            },
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform(), useful in exploratory notebooks."""
        return self.transform(df)

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        prof = self.transform(df)
        grouped = prof.groupby("capital_regime", sort=False)
        summary = grouped.agg(
            observations=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            mean_ticket_size=("ticket_size_proxy", "mean"),
            mean_vwap_deviation=("vwap_deviation", "mean"),
            mean_volatility=("realized_volatility", "mean"),
            mean_impact=("impact_proxy", "mean"),
        )
        return summary.reindex(
            [x for x in self.labels if x in summary.index]
        )
