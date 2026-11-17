"""Formal non-predictive Mathematical Radiography synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def _number(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(value: Any, digits: int = 4) -> str:
    v = _number(value)
    return f"{v:.{digits}g}" if v is not None else "unknown"


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


@dataclass
class RadiografiaMatematica:
    dataset_id: str
    scale_s: str
    ambient_space_dim_j: int | None
    intrinsic_dimension: float | None
    dominant_geometry_x: str
    relative_to_flat: float | None
    structural_properties: dict[str, Any]
    cascade_and_fingerprint_summary: dict[str, Any] | None = None
    ecosystem_couplings: list[dict[str, Any]] | None = None
    formal_statements_es: list[str] = field(default_factory=list)
    formal_statements_en: list[str] = field(default_factory=list)

    @property
    def statements_es(self) -> list[str]:
        return self.formal_statements_es

    @property
    def statements_en(self) -> list[str]:
        return self.formal_statements_en

    def to_dict(self) -> dict[str, Any]:
        payload = _safe(asdict(self))
        payload["scientific_statements"] = {
            "es": list(self.formal_statements_es),
            "en": list(self.formal_statements_en),
        }
        return payload

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    def to_markdown(self) -> str:
        lines = [
            f"# Radiografia Matematica: {self.dataset_id}",
            "",
            f"- Escala S: {self.scale_s}",
            f"- Espacio ambiente J: {self.ambient_space_dim_j if self.ambient_space_dim_j is not None else 'unknown'}",
            f"- Dimension intrinseca: {_fmt(self.intrinsic_dimension)}",
            f"- Geometria dominante X: {self.dominant_geometry_x}",
            f"- Relative to flat: {_fmt(self.relative_to_flat)}",
            "",
            "## Proposiciones estructurales",
        ]
        lines.extend(f"- {text}" for text in self.formal_statements_es)
        if self.formal_statements_en:
            lines += ["", "## Structural propositions"]
            lines.extend(f"- {text}" for text in self.formal_statements_en)
        return "\n".join(lines)


class MathematicalRadiographer:
    """Convert ESO exploration/diagnosis output into formal structural claims."""

    def __init__(self, language: str = "both") -> None:
        if language not in {"es", "en", "both"}:
            raise ValueError("language must be 'es', 'en', or 'both'")
        self.language = language

    @staticmethod
    def _couplings(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if hasattr(value, "to_dict") and not isinstance(value, Mapping):
            value = value.to_dict()
        if isinstance(value, Mapping):
            for key in ("findings", "edges", "ecosystem_couplings"):
                if key in value:
                    value = value[key]
                    break
            else:
                value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("ecosystem_couplings must be a sequence, mapping, or analyzer result")
        return [_safe(dict(item)) for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _scale(report: Mapping[str, Any], explicit: str | None) -> str:
        if explicit is not None:
            return str(explicit)
        candidates = [
            report.get("scale_s"),
            _nested(report, "dataset", "scale_s"),
            _nested(report, "dataset", "frequency"),
            _nested(report, "dataset", "timeframe"),
            _nested(report, "config", "scale_s"),
        ]
        return str(next((x for x in candidates if x is not None), "unspecified"))

    @staticmethod
    def _properties(diagnosis: Mapping[str, Any], best: Mapping[str, Any]) -> dict[str, Any]:
        stationarity = diagnosis.get("stationarity", {})
        symmetries = diagnosis.get("symmetries", {})
        lyapunov = diagnosis.get("lyapunov", {})
        return {
            "stationarity": {
                "label": stationarity.get("label", "unknown") if isinstance(stationarity, Mapping) else "unknown",
                "details": _safe(stationarity),
            },
            "periodicity": {
                "periodic_hint": bool(symmetries.get("periodic_hint", False)) if isinstance(symmetries, Mapping) else False,
                "dominant_frequency": _safe(symmetries.get("dominant_frequency")) if isinstance(symmetries, Mapping) else None,
            },
            "chaos": {
                "lyapunov": _safe(lyapunov.get("value")) if isinstance(lyapunov, Mapping) else None,
                "chaotic_hint": bool(lyapunov.get("chaotic_hint", False)) if isinstance(lyapunov, Mapping) else False,
            },
            "topology": {
                "smoothness": _safe(best.get("smoothness")),
                "utilization": _safe(best.get("latent_utilization", best.get("utilization"))),
                "reconstruction_error": _safe(best.get("reconstruction_error")),
            },
        }

    @staticmethod
    def _geometry_statement(
        scale: str,
        ambient: int | None,
        intrinsic: float | None,
        geometry: str,
        relative: float | None,
        props: Mapping[str, Any],
    ) -> tuple[str, str]:
        stationarity = _nested(props, "stationarity", "label", default="unknown")
        periodic = _nested(props, "periodicity", "periodic_hint", default=False)
        frequency = _nested(props, "periodicity", "dominant_frequency")
        if isinstance(frequency, Mapping):
            frequency = frequency.get("frequency")
        chaotic = _nested(props, "chaos", "chaotic_hint", default=False)
        lyapunov = _nested(props, "chaos", "lyapunov")
        smoothness = _nested(props, "topology", "smoothness")
        utilization = _nested(props, "topology", "utilization")
        j = str(ambient) if ambient is not None else "unknown"
        rpk = (
            f"R=stationarity:{stationarity}, P=periodic:{bool(periodic)}/f:{_fmt(frequency)}, "
            f"K=chaotic:{bool(chaotic)}/lyapunov:{_fmt(lyapunov)}"
        )
        m = f"intrinsic_dimension={_fmt(intrinsic)}"
        es = (
            f"Los datos a una escala {scale} modelados en un espacio {j}-dimensional con propiedades {rpk} "
            f"revelan una Geometria {geometry} con caracteristicas {m} "
            f"(relative_to_flat={_fmt(relative)}, smoothness={_fmt(smoothness)}, utilization={_fmt(utilization)})."
        )
        en = (
            f"Data at scale {scale} modeled in a {j}-dimensional ambient space with properties {rpk} "
            f"reveal Geometry {geometry} with characteristics {m} "
            f"(relative_to_flat={_fmt(relative)}, smoothness={_fmt(smoothness)}, utilization={_fmt(utilization)})."
        )
        return es, en

    @staticmethod
    def _coupling_statement(coupling: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        source = str(coupling.get("source", coupling.get("indicator", coupling.get("x", "X"))))
        target = str(coupling.get("target", coupling.get("y", "Y")))
        r = _number(coupling.get("peak_corr", coupling.get("correlation", coupling.get("r"))))
        if r is None:
            return [], []
        periods = coupling.get("periods")
        if not periods:
            start, end = coupling.get("period_start"), coupling.get("period_end")
            periods = [f"{start}..{end}"] if start is not None or end is not None else ["full sample"]
        period_text = ", ".join(str(p) for p in periods)
        stability = coupling.get("window_stability", coupling.get("stability", 0.0))
        regime = str(coupling.get("regime", "active"))
        lag = coupling.get("best_lag", coupling.get("lag", 0))
        sign = str(coupling.get("sign", "positive" if r >= 0 else "inverse"))
        sign_es = "positiva" if sign in {"positive", "positiva", "+"} else "inversa"
        sign_en = "positive" if sign_es == "positiva" else "inverse"

        if regime == "breakdown":
            return [
                (
                    f"El acoplamiento entre {source} y {target} se debilita o rompe en los periodos {period_text} "
                    f"(r={_fmt(r)}, estabilidad={_fmt(stability)}); no se interpreta como una prediccion."
                )
            ], [
                (
                    f"Coupling between {source} and {target} weakens or breaks down during periods {period_text} "
                    f"(r={_fmt(r)}, stability={_fmt(stability)}); this is not interpreted as a prediction."
                )
            ]

        es = [
            (
                f"Al parecer hay una correlacion {sign_es} entre {source} y {target} con rezago {lag} "
                f"en los periodos {period_text} (r={_fmt(r)}, estabilidad={_fmt(stability)})."
            ),
            (
                f"Existe una correlacion entre {source} y el volumen/flujo de {target} en los periodos {period_text} "
                f"con signo {sign_es}, rezago {lag} y r={_fmt(r)}; se describe como acoplamiento estructural, no como prediccion."
            ),
        ]
        en = [
            (
                f"There appears to be a {sign_en} correlation between {source} and {target} at lag {lag} "
                f"during periods {period_text} (r={_fmt(r)}, stability={_fmt(stability)})."
            )
        ]
        return es, en

    def synthesize(
        self,
        exploration_report: Mapping[str, Any],
        scale_s: str | None = None,
        ecosystem_couplings: Any = None,
        cascade_and_fingerprint_summary: Mapping[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> RadiografiaMatematica:
        if not isinstance(exploration_report, Mapping):
            raise TypeError("exploration_report must be a mapping")
        diagnosis = exploration_report.get("diagnosis", exploration_report)
        diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
        best = exploration_report.get("best") or {}
        if not isinstance(best, Mapping):
            best = {}
        if not best:
            evaluations = exploration_report.get("evaluations")
            if isinstance(evaluations, Sequence) and evaluations and isinstance(evaluations[0], Mapping):
                best = evaluations[0]

        dataset = str(dataset_id or exploration_report.get("dataset_id", "anonymous"))
        scale = self._scale(exploration_report, scale_s)
        ambient_value = best.get("ambient_dim", exploration_report.get("ambient_space_dim_j"))
        ambient_num = _number(ambient_value)
        ambient = int(ambient_num) if ambient_num is not None else None
        intrinsic = _number(_nested(diagnosis, "dimension", "consensus_dimension"))
        geometry = str(best.get("manifold", exploration_report.get("dominant_geometry_x", "unknown")))
        relative = _number(best.get("relative_to_flat", exploration_report.get("relative_to_flat")))
        props = self._properties(diagnosis, best)
        couplings = self._couplings(ecosystem_couplings)

        geometry_es, geometry_en = self._geometry_statement(
            scale, ambient, intrinsic, geometry, relative, props
        )
        es = [geometry_es] if self.language in {"es", "both"} else []
        en = [geometry_en] if self.language in {"en", "both"} else []
        for coupling in couplings:
            es_items, en_items = self._coupling_statement(coupling)
            if self.language in {"es", "both"}:
                es.extend(es_items)
            if self.language in {"en", "both"}:
                en.extend(en_items)

        return RadiografiaMatematica(
            dataset_id=dataset,
            scale_s=scale,
            ambient_space_dim_j=ambient,
            intrinsic_dimension=intrinsic,
            dominant_geometry_x=geometry,
            relative_to_flat=relative,
            structural_properties=props,
            cascade_and_fingerprint_summary=(
                _safe(dict(cascade_and_fingerprint_summary))
                if cascade_and_fingerprint_summary is not None else None
            ),
            ecosystem_couplings=couplings or None,
            formal_statements_es=es,
            formal_statements_en=en,
        )

    def generate(self, *args: Any, **kwargs: Any) -> RadiografiaMatematica:
        return self.synthesize(*args, **kwargs)

    def radiograph(self, *args: Any, **kwargs: Any) -> RadiografiaMatematica:
        return self.synthesize(*args, **kwargs)
