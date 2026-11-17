"""Directional lead-lag and exogenous coupling maps for ESO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _pair(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return A[t], B[t+lag]; positive lag means A leads B."""
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        k = -lag
        x, y = a[k:], b[:-k]
    else:
        x, y = a, b
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _corr(x: np.ndarray, y: np.ndarray, minimum: int) -> float:
    if len(x) < minimum or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best(a: np.ndarray, b: np.ndarray, lags: Iterable[int], minimum: int) -> tuple[int, float]:
    lag0, r0 = 0, float("nan")
    for lag in lags:
        r = _corr(*_pair(a, b, int(lag)), minimum)
        if np.isfinite(r) and (not np.isfinite(r0) or abs(r) > abs(r0)):
            lag0, r0 = int(lag), float(r)
    return lag0, r0


def _spearman(x: np.ndarray, y: np.ndarray, minimum: int) -> float:
    if len(x) < minimum:
        return float("nan")
    r = spearmanr(x, y, nan_policy="omit").statistic
    return float(r) if np.isfinite(r) else float("nan")


def _mi(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    if len(x) < 4:
        return float("nan")
    bins = max(2, min(int(bins), max(2, int(np.sqrt(len(x))))))
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    if hist.sum() == 0:
        return float("nan")
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    return value


def _label(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value)


@dataclass(frozen=True)
class InfluenceEdge:
    source: str
    target: str
    direction: str
    sign: str
    best_lag: int
    peak_corr: float
    strength: float
    window_stability: float
    spearman_corr: float
    mutual_information: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityInfluenceMapper:
    """Extract uni-/bi-directional influence edges from a numeric DataFrame."""

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        max_lag: int = 8,
        threshold: float = 0.55,
        dominance_ratio: float = 1.20,
        window_size: int | None = None,
        window_step: int | None = None,
        min_samples: int = 12,
        mi_bins: int = 10,
    ) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if dominance_ratio < 1:
            raise ValueError("dominance_ratio must be >= 1")
        self.data = data
        self.max_lag = int(max_lag)
        self.threshold = float(threshold)
        self.dominance_ratio = float(dominance_ratio)
        self.window_size = window_size
        self.window_step = window_step
        self.min_samples = int(min_samples)
        self.mi_bins = int(mi_bins)
        self.edges_: list[dict[str, Any]] = []
        self.relationships_: list[dict[str, Any]] = []
        self.columns_: list[str] = []

    def lag_profile(self, a: pd.Series, b: pd.Series) -> list[dict[str, Any]]:
        av, bv = np.asarray(a, float), np.asarray(b, float)
        return [
            {"lag": lag, "corr": _corr(*_pair(av, bv, lag), self.min_samples)}
            for lag in range(-self.max_lag, self.max_lag + 1)
        ]

    def _stability(self, a: np.ndarray, b: np.ndarray, lag: int, reference: float) -> float:
        x, y = _pair(a, b, lag)
        n = len(x)
        if n < self.min_samples or not np.isfinite(reference):
            return 0.0
        window = min(n, int(self.window_size or max(self.min_samples, max(24, n // 4))))
        step = int(self.window_step or max(1, window // 2))
        rs = []
        for start in range(0, n - window + 1, step):
            r = _corr(x[start:start + window], y[start:start + window], self.min_samples)
            if np.isfinite(r):
                rs.append(r)
        if not rs:
            r = _corr(x, y, self.min_samples)
            rs = [r] if np.isfinite(r) else []
        if not rs:
            return 0.0
        wanted_sign = 1 if reference >= 0 else -1
        good = sum(np.sign(r) == wanted_sign and abs(r) >= 0.8 * self.threshold for r in rs)
        return float(good / len(rs))

    def _edge(
        self, source: str, target: str, direction: str, lag: int, r: float,
        source_values: np.ndarray, target_values: np.ndarray,
    ) -> dict[str, Any]:
        x, y = _pair(source_values, target_values, lag)
        return InfluenceEdge(
            source=source,
            target=target,
            direction=direction,
            sign="positive" if r >= 0 else "inverse",
            best_lag=abs(int(lag)),
            peak_corr=float(r),
            strength=float(abs(r)),
            window_stability=self._stability(source_values, target_values, lag, r),
            spearman_corr=_spearman(x, y, self.min_samples),
            mutual_information=_mi(x, y, self.mi_bins),
        ).to_dict()

    def analyze_pair(
        self, a: pd.Series, b: pd.Series,
        name_a: str | None = None, name_b: str | None = None,
    ) -> dict[str, Any]:
        na = str(name_a if name_a is not None else (a.name if a.name is not None else "A"))
        nb = str(name_b if name_b is not None else (b.name if b.name is not None else "B"))
        av, bv = np.asarray(a, float), np.asarray(b, float)
        neg_lag, neg_r = _best(av, bv, range(-self.max_lag, 0), self.min_samples)
        pos_lag, pos_r = _best(av, bv, range(1, self.max_lag + 1), self.min_samples)
        neg_s = abs(neg_r) if np.isfinite(neg_r) else 0.0
        pos_s = abs(pos_r) if np.isfinite(pos_r) else 0.0
        neg_on, pos_on = neg_s >= self.threshold, pos_s >= self.threshold

        if pos_on and neg_on:
            ratio = max(pos_s, neg_s) / max(min(pos_s, neg_s), 1e-12)
            if ratio <= self.dominance_ratio:
                relation = "bidirectional"
            else:
                relation = "unidirectional_a_to_b" if pos_s > neg_s else "unidirectional_b_to_a"
        elif pos_on:
            relation = "unidirectional_a_to_b"
        elif neg_on:
            relation = "unidirectional_b_to_a"
        else:
            relation = "none"

        edges = []
        if relation in {"unidirectional_a_to_b", "bidirectional"}:
            edges.append(self._edge(na, nb, relation, pos_lag, pos_r, av, bv))
        if relation in {"unidirectional_b_to_a", "bidirectional"}:
            edges.append(self._edge(nb, na, relation, abs(neg_lag), neg_r, bv, av))
        return {
            "entity_a": na,
            "entity_b": nb,
            "relationship": relation,
            "a_to_b": {"lag": int(pos_lag), "peak_corr": float(pos_r), "strength": float(pos_s)},
            "b_to_a": {"lag": abs(int(neg_lag)), "peak_corr": float(neg_r), "strength": float(neg_s)},
            "zero_lag_corr": _corr(*_pair(av, bv, 0), self.min_samples),
            "edges": edges,
            "lag_profile": self.lag_profile(a, b),
        }

    def fit(self, data: pd.DataFrame | None = None) -> "EntityInfluenceMapper":
        frame = data if data is not None else self.data
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if frame.shape[1] < 2:
            raise ValueError("data must contain at least two entity columns")
        frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.data = frame
        self.columns_ = [str(c) for c in frame.columns]
        self.edges_, self.relationships_ = [], []
        for i, a in enumerate(frame.columns):
            for b in frame.columns[i + 1:]:
                result = self.analyze_pair(frame[a], frame[b], str(a), str(b))
                self.relationships_.append(result)
                self.edges_.extend(result["edges"])
        return self

    def map_influences(self, data: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        return list(self.fit(data).edges_)

    def fit_transform(self, data: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        return self.map_influences(data)

    def to_adjacency(self) -> dict[str, list[dict[str, Any]]]:
        adjacency = {name: [] for name in self.columns_}
        for edge in self.edges_:
            adjacency.setdefault(edge["source"], []).append(dict(edge))
            adjacency.setdefault(edge["target"], [])
        return adjacency

    def to_dict(self) -> dict[str, Any]:
        return _safe({
            "nodes": list(self.columns_),
            "edges": list(self.edges_),
            "relationships": list(self.relationships_),
            "adjacency": self.to_adjacency(),
            "lag_convention": "corr(source[t], target[t + lag]); positive lag means source leads target",
        })

    def build_graph(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        if data is not None or not self.relationships_:
            self.fit(data)
        return self.to_dict()

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class ExogenousCouplingAnalyzer:
    """Find local exogenous coupling regimes even when global correlation is weak."""

    def __init__(
        self,
        max_lag: int = 8,
        threshold: float = 0.55,
        window_size: int = 48,
        step: int | None = None,
        min_samples: int = 12,
        mi_bins: int = 10,
    ) -> None:
        if max_lag < 0 or window_size < 3:
            raise ValueError("max_lag must be >= 0 and window_size >= 3")
        self.max_lag = int(max_lag)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.step = int(step or max(1, window_size // 4))
        self.min_samples = int(min_samples)
        self.mi_bins = int(mi_bins)
        self.findings_: list[dict[str, Any]] = []
        self.window_regimes_: list[dict[str, Any]] = []

    @staticmethod
    def _frame(exogenous: pd.DataFrame | Mapping[str, Any] | pd.Series) -> pd.DataFrame:
        if isinstance(exogenous, pd.Series):
            return exogenous.to_frame(exogenous.name or "exogenous")
        if isinstance(exogenous, pd.DataFrame):
            return exogenous.copy()
        if isinstance(exogenous, Mapping):
            return pd.DataFrame(exogenous)
        raise TypeError("exogenous must be a Series, DataFrame, or mapping")

    def _window(self, source: str, target: str, frame: pd.DataFrame) -> dict[str, Any]:
        a, b = frame[source].to_numpy(float), frame[target].to_numpy(float)
        lag, r = _best(a, b, range(-self.max_lag, self.max_lag + 1), self.min_samples)
        x, y = _pair(a, b, lag)
        return {
            "indicator": source,
            "target": target,
            "best_lag": int(lag),
            "peak_corr": float(r),
            "strength": float(abs(r)) if np.isfinite(r) else 0.0,
            "sign": "positive" if np.isfinite(r) and r >= 0 else "inverse",
            "active": bool(np.isfinite(r) and abs(r) >= self.threshold),
            "spearman_corr": _spearman(x, y, self.min_samples),
            "mutual_information": _mi(x, y, self.mi_bins),
        }

    def _findings(self, windows: list[dict[str, Any]], global_corr: float) -> list[dict[str, Any]]:
        findings = []
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_key = None
        for item in windows:
            key = (item["sign"], int(np.sign(item["best_lag"]))) if item["active"] else None
            if item["active"] and (not current or key == current_key):
                current.append(item)
                current_key = key
            else:
                if current:
                    groups.append(current)
                    current = []
                current_key = None
                if item["active"]:
                    current = [item]
                    current_key = key
        if current:
            groups.append(current)

        for group in groups:
            rep = max(group, key=lambda item: item["strength"])
            mean_r = float(np.mean([item["peak_corr"] for item in group]))
            if np.isfinite(global_corr) and abs(global_corr) >= 0.15 and np.sign(mean_r) != np.sign(global_corr):
                regime = "sign_reversal"
            elif not np.isfinite(global_corr) or abs(mean_r) >= abs(global_corr) + 0.10:
                regime = "strengthened"
            else:
                regime = "active"
            item = dict(rep)
            item.update({
                "period_start": group[0]["period_start"],
                "period_end": group[-1]["period_end"],
                "periods": [f"{group[0]['period_start']}..{group[-1]['period_end']}"],
                "peak_corr": mean_r,
                "strength": abs(mean_r),
                "window_stability": len(group) / max(1, len(windows)),
                "active_windows": len(group),
                "total_windows": len(windows),
                "global_corr": float(global_corr),
                "regime": regime,
                "direction": (
                    f"{rep['indicator']}_to_{rep['target']}" if rep["best_lag"] > 0
                    else f"{rep['target']}_to_{rep['indicator']}" if rep["best_lag"] < 0
                    else "contemporaneous"
                ),
            })
            findings.append(item)

        active = {i for i, item in enumerate(windows) if item["active"]}
        i = 0
        while i < len(windows):
            if i in active:
                i += 1
                continue
            j = i
            while j + 1 < len(windows) and j + 1 not in active:
                j += 1
            if i - 1 in active or j + 1 in active:
                group = windows[i:j + 1]
                finite = [item["peak_corr"] for item in group if np.isfinite(item["peak_corr"])]
                if finite:
                    rep = max(group, key=lambda item: item["strength"])
                    mean_r = float(np.mean(finite))
                    item = dict(rep)
                    item.update({
                        "period_start": group[0]["period_start"],
                        "period_end": group[-1]["period_end"],
                        "periods": [f"{group[0]['period_start']}..{group[-1]['period_end']}"],
                        "peak_corr": mean_r,
                        "strength": abs(mean_r),
                        "window_stability": 0.0,
                        "active_windows": 0,
                        "total_windows": len(windows),
                        "global_corr": float(global_corr),
                        "regime": "breakdown",
                        "direction": "none",
                    })
                    findings.append(item)
            i = j + 1
        return findings

    def analyze(
        self,
        target: pd.Series,
        exogenous: pd.DataFrame | Mapping[str, Any] | pd.Series,
        target_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(target, pd.Series):
            target = pd.Series(target)
        target_name = str(target_name or target.name or "target")
        target = pd.to_numeric(target, errors="coerce").rename(target_name)
        exog = self._frame(exogenous).apply(pd.to_numeric, errors="coerce")
        self.findings_, self.window_regimes_ = [], []

        for column in exog.columns:
            source = str(column)
            frame = pd.concat([exog[column].rename(source), target], axis=1, join="inner").dropna()
            if len(frame) < self.min_samples:
                continue
            global_r = _corr(frame[source].to_numpy(float), frame[target_name].to_numpy(float), self.min_samples)
            window = min(self.window_size, len(frame))
            starts = list(range(0, len(frame) - window + 1, self.step))
            if starts and starts[-1] != len(frame) - window:
                starts.append(len(frame) - window)
            windows = []
            for start in starts:
                record = self._window(source, target_name, frame.iloc[start:start + window])
                record["period_start"] = _label(frame.index[start])
                record["period_end"] = _label(frame.index[start + window - 1])
                record["global_corr"] = float(global_r)
                windows.append(record)
            self.window_regimes_.extend(windows)
            self.findings_.extend(self._findings(windows, global_r))
        return list(self.findings_)

    def analyze_couplings(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self.analyze(*args, **kwargs)

    def detect_regimes(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self.analyze(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return _safe({"findings": self.findings_, "windows": self.window_regimes_})

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)
