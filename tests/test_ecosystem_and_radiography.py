from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eso.radiography import (
    EntityInfluenceMapper,
    ExogenousCouplingAnalyzer,
    MathematicalRadiographer,
)


def test_entity_mapper_detects_direction_inverse_and_independence():
    rng = np.random.default_rng(123)
    n = 360
    x = rng.normal(size=n)
    y = rng.normal(scale=0.03, size=n)
    y[2:] += x[:-2]
    inv = rng.normal(scale=0.03, size=n)
    inv[3:] -= x[:-3]
    noise = rng.normal(size=n)
    frame = pd.DataFrame({"X": x, "Y": y, "INV": inv, "NOISE": noise})

    mapper = EntityInfluenceMapper(
        max_lag=5, threshold=0.75, dominance_ratio=1.15, window_size=80
    ).fit(frame)

    xy = next(
        item for item in mapper.relationships_
        if item["entity_a"] == "X" and item["entity_b"] == "Y"
    )
    assert xy["relationship"] == "unidirectional_a_to_b"
    assert xy["edges"][0]["source"] == "X"
    assert xy["edges"][0]["target"] == "Y"
    assert xy["edges"][0]["best_lag"] == 2
    assert xy["edges"][0]["sign"] == "positive"
    assert xy["edges"][0]["window_stability"] >= 0.8
    assert xy["edges"][0]["mutual_information"] > 0

    xinv = next(
        item for item in mapper.relationships_
        if item["entity_a"] == "X" and item["entity_b"] == "INV"
    )
    assert xinv["relationship"] == "unidirectional_a_to_b"
    assert xinv["edges"][0]["best_lag"] == 3
    assert xinv["edges"][0]["sign"] == "inverse"

    xnoise = next(
        item for item in mapper.relationships_
        if item["entity_a"] == "X" and item["entity_b"] == "NOISE"
    )
    assert xnoise["relationship"] == "none"
    assert set(mapper.to_adjacency()) == set(frame.columns)
    json.dumps(mapper.to_dict(), allow_nan=False)


def test_entity_mapper_detects_bidirectional_oscillatory_coupling():
    t = np.arange(480)
    x = np.sin(2 * np.pi * t / 31.0) + 0.25 * np.sin(2 * np.pi * t / 13.0)
    y = 0.5 * np.roll(x, 2) + 0.5 * np.roll(x, -2)
    mapper = EntityInfluenceMapper(
        max_lag=5, threshold=0.75, dominance_ratio=1.10, window_size=100
    )
    relation = mapper.analyze_pair(pd.Series(x, name="A"), pd.Series(y, name="B"))
    assert relation["relationship"] == "bidirectional"
    assert {edge["source"] for edge in relation["edges"]} == {"A", "B"}


def test_exogenous_analyzer_finds_local_regime_when_global_signal_is_diluted():
    rng = np.random.default_rng(7)
    n = 320
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    exog = rng.normal(size=n)
    target = rng.normal(size=n)
    target[112:212] = exog[110:210] + rng.normal(scale=0.05, size=100)

    analyzer = ExogenousCouplingAnalyzer(
        max_lag=4, threshold=0.75, window_size=60, step=10
    )
    findings = analyzer.analyze(
        pd.Series(target, index=idx, name="BTC_flow"),
        pd.DataFrame({"macro_index": exog}, index=idx),
    )

    active = [
        finding for finding in findings
        if finding["indicator"] == "macro_index" and finding["regime"] != "breakdown"
    ]
    assert active
    best = max(active, key=lambda finding: finding["strength"])
    assert best["sign"] == "positive"
    assert best["best_lag"] == 2
    assert best["strength"] > 0.85
    assert abs(best["global_corr"]) < best["strength"]
    assert best["periods"]
    assert best["regime"] in {"strengthened", "active"}


def test_exogenous_analyzer_marks_breakdown_after_active_regime():
    rng = np.random.default_rng(99)
    n = 260
    exog = rng.normal(size=n)
    target = rng.normal(size=n)
    target[50:150] = exog[49:149] + rng.normal(scale=0.04, size=100)
    analyzer = ExogenousCouplingAnalyzer(
        max_lag=3, threshold=0.72, window_size=50, step=10
    )
    findings = analyzer.analyze(
        pd.Series(target, name="flow"),
        pd.DataFrame({"climate": exog}),
    )
    assert any(item["regime"] == "breakdown" for item in findings)


def test_mathematical_radiographer_synthesizes_serializable_formal_propositions():
    report = {
        "dataset_id": "btc-4h",
        "diagnosis": {
            "dimension": {"consensus_dimension": 2.4},
            "stationarity": {"label": "difference_stationary"},
            "symmetries": {
                "periodic_hint": True,
                "dominant_frequency": {"frequency": 0.125},
            },
            "lyapunov": {"value": 0.031, "chaotic_hint": True},
        },
        "best": {
            "manifold": "torus2",
            "ambient_dim": 3,
            "relative_to_flat": 0.82,
            "smoothness": 0.14,
            "latent_utilization": 0.76,
            "reconstruction_error": 0.09,
        },
    }
    couplings = [{
        "indicator": "macro_index",
        "target": "BTC_flow",
        "sign": "positive",
        "best_lag": 2,
        "peak_corr": 0.91,
        "window_stability": 0.8,
        "periods": ["2026-02-01..2026-02-10"],
    }]

    radiography = MathematicalRadiographer().synthesize(
        report,
        scale_s="4h",
        ecosystem_couplings=couplings,
        cascade_and_fingerprint_summary={"eta": np.float64(0.63)},
    )
    payload = radiography.to_dict()
    json.dumps(payload, allow_nan=False)

    assert payload["scale_s"] == "4h"
    assert payload["ambient_space_dim_j"] == 3
    assert payload["dominant_geometry_x"] == "torus2"
    geometry = radiography.formal_statements_es[0]
    assert "Los datos a una escala 4h modelados en un espacio 3-dimensional" in geometry
    assert "revelan una Geometria torus2" in geometry
    assert "relative_to_flat=0.82" in geometry
    assert any(
        "Al parecer hay una correlacion positiva entre macro_index y BTC_flow con rezago 2" in text
        and "2026-02-01..2026-02-10" in text
        for text in radiography.formal_statements_es
    )
    assert any(
        "Existe una correlacion entre macro_index y el volumen/flujo de BTC_flow" in text
        for text in radiography.formal_statements_es
    )
    assert "# Radiografia Matematica: btc-4h" in radiography.to_markdown()
