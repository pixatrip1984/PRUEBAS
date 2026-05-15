from eso.reporting import build_model_handoff


def _report(feature_mode="compact"):
    return {
        "dataset_id": "btc_test",
        "dataset": {
            "columns": ["log_return", "vol_20", "vwap_dev", "volume_imbalance"],
            "shape": [1000, 4],
            "normalized": "robust",
        },
        "feature_mode": feature_mode,
        "projection_method": "umap",
        "config": {"k": 8, "n_masks": 5},
        "diagnosis": {"dimension": {"consensus_dimension": 4}},
        "best": {
            "rank": 1,
            "manifold": "cylinder",
            "score_mean": 0.12,
            "reconstruction_error_mean": 0.06,
            "reconstruction_error_std": 0.01,
            "smoothness": 0.4,
            "latent_utilization": 0.8,
        },
        "evaluations": [
            {"rank": 1, "manifold": "cylinder", "score_mean": 0.12, "reconstruction_error_mean": 0.06},
            {"rank": 2, "manifold": "torus2", "score_mean": 0.18, "reconstruction_error_mean": 0.11},
        ],
    }


def test_model_handoff_has_schema_and_topology():
    handoff = build_model_handoff(_report())
    assert handoff["schema_version"] == "eso.model_handoff.v1"
    assert handoff["topology"]["best"]["manifold"] == "cylinder"
    assert handoff["topology"]["evaluations_ranked"][0]["rank"] == 1


def test_model_handoff_prioritizes_volatility_for_market_features():
    handoff = build_model_handoff(_report())
    tasks = {t["name"]: t for t in handoff["recommended_model_tasks"]}
    assert tasks["future_realized_volatility"]["priority"] == "high"
    assert tasks["volatility_regime"]["priority"] == "high"
    assert "vol_20" in handoff["feature_contract"]["known_feature_descriptions"]


def test_model_handoff_records_causal_guardrails_and_known_nulls():
    handoff = build_model_handoff(_report())
    guardrails = " ".join(handoff["causal_guardrails"])
    assert "center=False" in guardrails
    assert "dropna" in guardrails
    assert any(item["status"] == "rejected" for item in handoff["known_nulls"])
    assert any("model-eval" in command for command in handoff["suggested_commands"])


def test_raw_generic_handoff_does_not_claim_btc_nulls():
    handoff = build_model_handoff(_report(feature_mode="raw"))
    assert handoff["known_nulls"] == []
