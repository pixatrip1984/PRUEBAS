"""Machine-readable handoff for downstream model-building agents."""

from __future__ import annotations

from copy import deepcopy


FINANCIAL_FEATURES = {
    "log_return": "One-bar log return; stationary price-change proxy.",
    "vol_20": "Rolling 20-bar standard deviation of log_return; causal volatility baseline.",
    "vwap_dev": "(close - vwap) / close; intrabar VWAP position when vwap is available.",
    "volume_imbalance": "(taker_buy_volume - taker_sell_volume) / volume; order-flow proxy.",
    "lr_z20": "log_return z-score over a 20-bar causal rolling window.",
    "ring_radius": "Distance from fitted 2D ring center; larger values mean stronger on-ring structure.",
    "sin_theta_24h": "Causal smoothed ring phase component over 24 bars.",
    "cos_theta_24h": "Causal smoothed ring phase component over 24 bars.",
}


def _best_summary(report: dict) -> dict:
    best = report.get("best") or {}
    return {
        "manifold": best.get("manifold"),
        "rank": best.get("rank"),
        "score": best.get("score_mean", best.get("score")),
        "reconstruction_error": best.get(
            "reconstruction_error_mean", best.get("reconstruction_error")
        ),
        "reconstruction_error_std": best.get("reconstruction_error_std"),
        "smoothness": best.get("smoothness"),
        "latent_utilization": best.get("latent_utilization"),
    }


def _topology_notes(report: dict) -> list[str]:
    best = report.get("best") or {}
    diagnosis = report.get("diagnosis", {})
    notes = []
    dim = diagnosis.get("dimension", {}).get("consensus_dimension")
    if dim is not None:
        notes.append(f"Intrinsic dimension estimate: {dim}.")
    manifold = best.get("manifold")
    if manifold:
        notes.append(f"Best candidate manifold: {manifold}.")
    if best.get("latent_utilization", 1.0) < 0.1:
        notes.append("Low latent utilization; avoid over-interpreting the winning manifold.")
    if best.get("reconstruction_error_std", 0.0) > abs(
        best.get("reconstruction_error_mean", best.get("reconstruction_error", 0.0)) or 0.0
    ):
        notes.append("Validation variance is high; rerun with more masks before model design.")
    return notes


def _feature_contract(report: dict) -> dict:
    dataset = report.get("dataset", {})
    feature_mode = report.get("feature_mode", "raw")
    columns = list(dataset.get("columns", []))
    known = {c: FINANCIAL_FEATURES[c] for c in columns if c in FINANCIAL_FEATURES}
    return {
        "feature_mode": feature_mode,
        "columns": columns,
        "known_feature_descriptions": known,
        "normalization": dataset.get("normalized"),
        "shape": dataset.get("shape"),
        "windowing": {
            "window_size": dataset.get("window_size"),
            "window_step": dataset.get("window_step"),
            "window_mode": dataset.get("window_mode"),
        },
    }


def _model_tasks(report: dict) -> list[dict]:
    feature_mode = report.get("feature_mode", "raw")
    columns = set(report.get("dataset", {}).get("columns", []))
    tasks: list[dict] = [
        {
            "name": "manifold_conditioned_autoencoder",
            "priority": "medium",
            "target": "reconstruct selected input features",
            "why": "Use the winning manifold as an inductive-bias hint, not as a supervised edge.",
            "baseline": "linear PCA reconstruction with the same latent dimension",
        }
    ]

    has_close = "close" in columns
    has_compact_market = {"log_return", "vol_20"}.issubset(columns) or feature_mode in {
        "compact",
        "returns_vol",
        "full",
    }
    has_micro = "volume_imbalance" in columns or feature_mode in {"compact", "microstructure", "full"}

    if has_close or has_compact_market:
        tasks.insert(
            0,
            {
                "name": "future_realized_volatility",
                "priority": "high",
                "target": "rv_h = std(log_return[t+1:t+h+1])",
                "suggested_horizons": [6, 12, 24],
                "baseline": "vol_20 persistence at t",
                "why": "BTC experiments found robust volatility signal; this is the first model-builder target to reproduce.",
            },
        )
        tasks.insert(
            1,
            {
                "name": "volatility_regime",
                "priority": "high",
                "target": "rv_h above training-set median",
                "suggested_horizons": [12],
                "baseline": "majority class and vol_20-only classifier",
                "why": "Useful as a strategy allocator even when direction is absent.",
            },
        )

    if has_micro:
        tasks.append(
            {
                "name": "short_horizon_direction_probe",
                "priority": "low",
                "target": "sign(log(close[t+h] / close[t]))",
                "suggested_horizons": [1, 2, 3],
                "baseline": "class balance and log_return/volume_imbalance logistic regression",
                "why": "BTC 12h direction failed; only short horizons remain plausible.",
            }
        )
    return tasks


def _known_nulls(report: dict) -> list[dict]:
    feature_mode = report.get("feature_mode", "raw")
    market_like = feature_mode in {"compact", "returns_vol", "microstructure", "full"}
    if not market_like:
        return []
    return [
        {
            "claim": "BTC 12h direction from current ring/order-flow features",
            "status": "rejected",
            "evidence": "Clean experiments cluster around 49-50% accuracy vs roughly 51% baseline.",
        },
        {
            "claim": "Centered rolling phase smoothing",
            "status": "invalid",
            "evidence": "center=True leaks future bars and created a false r_OOS=0.320 result.",
        },
    ]


def _guardrails() -> list[str]:
    return [
        "Fit scalers, embeddings, selectors, and model weights only on the training window.",
        "Use rolling(..., center=False) for every feature used at prediction time.",
        "After dropna(), align targets by original dataframe index, not by reset positions.",
        "Do not use future realized volatility as a regime filter.",
        "Report the majority-class baseline for classification and vol_20 persistence for volatility.",
        "Treat topology as a feature-generation hypothesis until a walk-forward test confirms value.",
    ]


def _suggested_commands(report: dict) -> list[str]:
    dataset = report.get("dataset", {})
    source = dataset.get("source_path", "<dataset.csv>")
    feature_mode = report.get("feature_mode", "raw")
    if source == "<dataframe>":
        source = "<dataset.csv>"
    return [
        (
            "python -m eso.cli model-eval "
            f"{source} --task volatility --feature-mode {feature_mode} "
            "--horizon 12 --output reports/model_volatility"
        ),
        (
            "python -m eso.cli model-eval "
            f"{source} --task regime --feature-mode {feature_mode} "
            "--horizon 12 --output reports/model_regime"
        ),
        (
            "python -m eso.cli model-eval "
            f"{source} --task direction --feature-mode {feature_mode} "
            "--horizon 3 --output reports/model_direction_h3"
        ),
    ]


def build_model_handoff(report: dict) -> dict:
    """Return a compact contract for agents that build downstream models.

    The handoff is intentionally conservative: it distinguishes validated
    local BTC findings from generic next experiments and includes guardrails
    for the specific lookahead bugs already found in this repo.
    """
    dataset = deepcopy(report.get("dataset", {}))
    return {
        "schema_version": "eso.model_handoff.v1",
        "dataset_id": report.get("dataset_id"),
        "dataset": dataset,
        "topology": {
            "best": _best_summary(report),
            "notes": _topology_notes(report),
            "evaluations_ranked": [
                {
                    "rank": e.get("rank"),
                    "manifold": e.get("manifold"),
                    "score": e.get("score_mean", e.get("score")),
                    "reconstruction_error": e.get(
                        "reconstruction_error_mean", e.get("reconstruction_error")
                    ),
                }
                for e in report.get("evaluations", [])
            ],
        },
        "feature_contract": _feature_contract(report),
        "recommended_model_tasks": _model_tasks(report),
        "known_nulls": _known_nulls(report),
        "causal_guardrails": _guardrails(),
        "suggested_commands": _suggested_commands(report),
        "reproducibility": {
            "eso_config": report.get("config", {}),
            "feature_mode": report.get("feature_mode", "raw"),
            "projection_method": report.get("projection_method", "linear"),
        },
    }
