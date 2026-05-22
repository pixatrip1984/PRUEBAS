import numpy as np
import pandas as pd

from eso.meta import STABLE, UNSTABLE, diagnose_association_stability
from eso.meta.stability import MIXED, score_correlation_stability


def test_wide_asset_dissect_correlations_get_stability_decisions():
    rows = pd.DataFrame([
        {
            "horizon": 3,
            "feature": "cycle_strength",
            "corr_rv_train": 0.12,
            "corr_rv_test": 0.10,
            "corr_lr_train": 0.08,
            "corr_lr_test": 0.01,
            "corr_direction_train": 0.09,
            "corr_direction_test": -0.10,
        }
    ])

    out = diagnose_association_stability(rows, min_abs_corr=0.05)
    decisions = out.set_index("target")["decision"].to_dict()

    assert decisions["rv"] == STABLE
    assert decisions["lr"] == MIXED
    assert decisions["direction"] == UNSTABLE
    assert out.loc[out["target"] == "rv", "stability_score"].item() == 1.0


def test_long_temporal_blocks_score_feature_target_pairs():
    rows = pd.DataFrame([
        {"horizon": 12, "feature": "x", "target": "future_rv", "block": "a", "corr": 0.07},
        {"horizon": 12, "feature": "x", "target": "future_rv", "block": "b", "corr": 0.08},
        {"horizon": 12, "feature": "x", "target": "future_rv", "block": "c", "corr": 0.09},
        {"horizon": 12, "feature": "x", "target": "future_lr", "block": "a", "corr": 0.07},
        {"horizon": 12, "feature": "x", "target": "future_lr", "block": "b", "corr": -0.08},
        {"horizon": 12, "feature": "x", "target": "future_lr", "block": "c", "corr": 0.06},
    ])

    out = diagnose_association_stability(rows, min_abs_corr=0.05, stable_sign_fraction=0.8)
    decisions = out.set_index("target")["decision"].to_dict()

    assert decisions["future_rv"] == STABLE
    assert decisions["future_lr"] == MIXED
    assert out.loc[out["target"] == "future_lr", "sign_consistency"].item() == 2 / 3


def test_missing_and_insufficient_blocks_are_not_stable():
    result = score_correlation_stability([0.20, np.nan], min_abs_corr=0.05, min_blocks=2)

    assert result["decision"] == UNSTABLE
    assert result["n_blocks"] == 1
    assert result["n_missing"] == 1
    assert result["stability_score"] < 1.0


def test_group_cols_can_keep_assets_separate_in_wide_rows():
    rows = pd.DataFrame([
        {"asset_id": "BTC", "horizon": 3, "feature": "f", "corr_rv_train": 0.10, "corr_rv_test": 0.11},
        {"asset_id": "ETH", "horizon": 3, "feature": "f", "corr_rv_train": 0.10, "corr_rv_test": -0.11},
    ])

    out = diagnose_association_stability(rows, min_abs_corr=0.05)
    decisions = out.set_index("asset_id")["decision"].to_dict()

    assert decisions == {"BTC": STABLE, "ETH": UNSTABLE}
