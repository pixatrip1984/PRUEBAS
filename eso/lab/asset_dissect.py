"""Asset dissection instruments for market datasets.

The goal is not to declare an edge. The goal is to expose enough structure for
model/trading agents to stop working blind: target distributions, feature-target
associations, quantile regimes, and ring phase sectors when available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eso.contracts import assert_valid_contract
from eso.signals.costs import DEFAULT_TAKER_ROUND_TRIP_COST, rate_to_bps
from eso.signals.targets import build_market_target_frame


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        value = obj.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _numeric_features(fv: pd.DataFrame) -> list[str]:
    exclude = {"close"}
    return [
        c for c in fv.select_dtypes(include=["number"]).columns
        if c not in exclude and not c.startswith("future_")
    ]


def _split_index(n: int, train_size: int | None, train_fraction: float) -> int:
    if train_size is None:
        train_size = int(n * train_fraction)
    return max(1, min(int(train_size), max(1, n - 1)))


def _corr(x: pd.Series, y: pd.Series) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 30:
        return float("nan")
    a = data.iloc[:, 0].to_numpy(dtype=float)
    b = data.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _target_summary(targets: pd.DataFrame, horizon: int, split: int, round_trip_cost_bps: float) -> dict:
    out = {"horizon": horizon}
    lr = targets[f"future_lr_{horizon}"]
    direction = targets[f"future_direction_{horizon}"]
    rv = targets[f"future_rv_{horizon}"]
    valid_dir = direction.dropna()
    valid_dir = valid_dir[valid_dir != 0]
    abs_lr_bps = lr.abs().dropna() * 10_000
    out.update({
        "n_lr": int(lr.notna().sum()),
        "future_lr_mean": float(lr.mean()),
        "future_lr_std": float(lr.std()),
        "future_rv_mean": float(rv.mean()),
        "future_rv_median": float(rv.median()),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "mean_abs_future_lr_bps": float(abs_lr_bps.mean()) if len(abs_lr_bps) else None,
        "median_abs_future_lr_bps": float(abs_lr_bps.median()) if len(abs_lr_bps) else None,
        "mean_abs_move_minus_cost_bps": (
            float(abs_lr_bps.mean() - round_trip_cost_bps) if len(abs_lr_bps) else None
        ),
        "cost_floor_cleared_rate": (
            float((abs_lr_bps > round_trip_cost_bps).mean()) if len(abs_lr_bps) else None
        ),
        "direction_up_rate": float((valid_dir > 0).mean()) if len(valid_dir) else None,
        "direction_baseline_accuracy": (
            float(max((valid_dir > 0).mean(), (valid_dir < 0).mean())) if len(valid_dir) else None
        ),
    })
    test_dir = valid_dir.loc[valid_dir.index >= split]
    if len(test_dir):
        out["test_direction_baseline_accuracy"] = float(max((test_dir > 0).mean(), (test_dir < 0).mean()))
        out["test_direction_up_rate"] = float((test_dir > 0).mean())
    return out


def _feature_summary(fv: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    rows = []
    for col in feature_cols:
        s = pd.to_numeric(fv[col], errors="coerce")
        rows.append({
            "feature": col,
            "missing_ratio": float(s.isna().mean()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    return rows


def _correlation_table(fv: pd.DataFrame, targets: pd.DataFrame, feature_cols: list[str], horizons: tuple[int, ...], split: int) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        for feature in feature_cols:
            x = fv[feature]
            rv = targets[f"future_rv_{horizon}"]
            lr = targets[f"future_lr_{horizon}"]
            direction = targets[f"future_direction_{horizon}"].replace(0, np.nan)
            rows.append({
                "horizon": horizon,
                "feature": feature,
                "corr_rv_all": _corr(x, rv),
                "corr_rv_train": _corr(x.iloc[:split], rv.iloc[:split]),
                "corr_rv_test": _corr(x.iloc[split:], rv.iloc[split:]),
                "corr_lr_all": _corr(x, lr),
                "corr_lr_train": _corr(x.iloc[:split], lr.iloc[:split]),
                "corr_lr_test": _corr(x.iloc[split:], lr.iloc[split:]),
                "corr_direction_all": _corr(x, direction),
                "corr_direction_train": _corr(x.iloc[:split], direction.iloc[:split]),
                "corr_direction_test": _corr(x.iloc[split:], direction.iloc[split:]),
            })
    out = pd.DataFrame(rows)
    out["max_abs_test_corr"] = out[["corr_rv_test", "corr_lr_test", "corr_direction_test"]].abs().max(axis=1)
    return out.sort_values(["horizon", "max_abs_test_corr"], ascending=[True, False])


def _edge_fields(mean_lr: float, lr_values: pd.Series, round_trip_cost_bps: float) -> dict:
    mean_lr_bps = float(mean_lr * 10_000)
    best_net_edge_bps = abs(mean_lr_bps) - float(round_trip_cost_bps)
    if best_net_edge_bps <= 0:
        best_direction = "NONE"
    else:
        best_direction = "LONG" if mean_lr_bps > 0 else "SHORT"
    abs_lr_bps = lr_values.abs().dropna() * 10_000
    return {
        "future_lr_mean_bps": mean_lr_bps,
        "best_direction_by_mean": best_direction,
        "best_net_edge_bps": float(best_net_edge_bps),
        "cost_floor_cleared_rate": (
            float((abs_lr_bps > round_trip_cost_bps).mean()) if len(abs_lr_bps) else None
        ),
    }


def _quantile_table(
    fv: pd.DataFrame,
    targets: pd.DataFrame,
    features: list[str],
    horizons: tuple[int, ...],
    round_trip_cost_bps: float,
    q: int = 5,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in fv.columns:
            continue
        values = pd.to_numeric(fv[feature], errors="coerce")
        try:
            bins = pd.qcut(values, q=q, duplicates="drop")
        except ValueError:
            continue
        for horizon in horizons:
            frame = pd.DataFrame({
                "bin": bins.astype(str),
                "feature_value": values,
                "future_rv": targets[f"future_rv_{horizon}"],
                "future_lr": targets[f"future_lr_{horizon}"],
                "direction": targets[f"future_direction_{horizon}"],
            }).dropna(subset=["bin", "future_rv", "future_lr"])
            for bin_name, grp in frame.groupby("bin", observed=True):
                direction = grp["direction"].replace(0, np.nan).dropna()
                row = {
                    "feature": feature,
                    "horizon": horizon,
                    "bin": bin_name,
                    "n": int(len(grp)),
                    "feature_min": float(grp["feature_value"].min()),
                    "feature_max": float(grp["feature_value"].max()),
                    "future_rv_mean": float(grp["future_rv"].mean()),
                    "future_lr_mean": float(grp["future_lr"].mean()),
                    "p_up": float((direction > 0).mean()) if len(direction) else None,
                    "round_trip_cost_bps": float(round_trip_cost_bps),
                }
                row.update(_edge_fields(row["future_lr_mean"], grp["future_lr"], round_trip_cost_bps))
                rows.append(row)
    return pd.DataFrame(rows)


def _phase_sector_table(
    fv: pd.DataFrame,
    targets: pd.DataFrame,
    horizons: tuple[int, ...],
    round_trip_cost_bps: float,
    phase_window: int = 24,
    sectors: int = 12,
) -> pd.DataFrame:
    sin_col = f"sin_theta_{phase_window}h"
    cos_col = f"cos_theta_{phase_window}h"
    if sin_col not in fv.columns or cos_col not in fv.columns:
        return pd.DataFrame()
    theta = np.mod(np.arctan2(fv[sin_col].to_numpy(dtype=float), fv[cos_col].to_numpy(dtype=float)), 2 * np.pi)
    sector = np.floor(theta / (2 * np.pi / sectors)).astype(int)
    rows = []
    for horizon in horizons:
        frame = pd.DataFrame({
            "sector": sector,
            "theta_deg": np.degrees(theta),
            "future_rv": targets[f"future_rv_{horizon}"],
            "future_lr": targets[f"future_lr_{horizon}"],
            "direction": targets[f"future_direction_{horizon}"],
        }).dropna(subset=["future_rv", "future_lr"])
        for sec, grp in frame.groupby("sector"):
            direction = grp["direction"].replace(0, np.nan).dropna()
            row = {
                "phase_window": phase_window,
                "horizon": horizon,
                "sector": int(sec),
                "sector_mid_deg": float((sec + 0.5) * 360.0 / sectors),
                "n": int(len(grp)),
                "theta_mean_deg": float(grp["theta_deg"].mean()),
                "future_rv_mean": float(grp["future_rv"].mean()),
                "future_lr_mean": float(grp["future_lr"].mean()),
                "p_up": float((direction > 0).mean()) if len(direction) else None,
                "round_trip_cost_bps": float(round_trip_cost_bps),
            }
            row.update(_edge_fields(row["future_lr_mean"], grp["future_lr"], round_trip_cost_bps))
            rows.append(row)
    return pd.DataFrame(rows)


def _default_quantile_features(feature_cols: list[str], corr_df: pd.DataFrame) -> list[str]:
    preferred = ["vol_20", "ring_radius", "lr_z20", "volume_imbalance", "log_return", "vwap_dev"]
    selected = [f for f in preferred if f in feature_cols]
    if not corr_df.empty:
        for feature in corr_df.sort_values("max_abs_test_corr", ascending=False)["feature"].tolist():
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= 8:
                break
    return selected


def dissect_feature_frame(
    fv: pd.DataFrame,
    asset_id: str = "asset",
    horizons: tuple[int, ...] = (3, 12),
    train_size: int | None = None,
    train_fraction: float = 0.7,
    phase_window: int = 24,
    round_trip_cost_bps: float = rate_to_bps(DEFAULT_TAKER_ROUND_TRIP_COST),
) -> dict:
    """Return asset-dissection tables and JSON summary."""
    if "close" not in fv.columns:
        raise ValueError("feature frame must include close")
    horizons = tuple(int(h) for h in horizons)
    round_trip_cost_bps = float(round_trip_cost_bps)
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be >= 0")
    targets = build_market_target_frame(fv, horizons=horizons, close_col="close")
    split = _split_index(len(fv), train_size, train_fraction)
    feature_cols = _numeric_features(fv)
    corr_df = _correlation_table(fv, targets, feature_cols, horizons, split)
    quantile_features = _default_quantile_features(feature_cols, corr_df)
    quantile_df = _quantile_table(fv, targets, quantile_features, horizons, round_trip_cost_bps)
    sector_df = _phase_sector_table(
        fv,
        targets,
        horizons,
        round_trip_cost_bps,
        phase_window=phase_window,
    )

    top_corr = corr_df.head(20).replace({np.nan: None}).to_dict(orient="records")
    result = {
        "schema_version": "eso.asset_dissect.v1",
        "asset_id": asset_id,
        "rows": int(len(fv)),
        "horizons": list(horizons),
        "split": {"train_size": split, "test_size": int(len(fv) - split), "shuffle": False},
        "cost_floor": {
            "round_trip_cost_bps": round_trip_cost_bps,
            "round_trip_cost_rate": round_trip_cost_bps / 10_000,
        },
        "features": feature_cols,
        "target_summary": [_target_summary(targets, h, split, round_trip_cost_bps) for h in horizons],
        "feature_summary": _feature_summary(fv, feature_cols),
        "top_feature_target_correlations": top_corr,
        "tables": {},
        "guardrails": [
            "Targets are indexed at decision time t.",
            "Train/test correlation columns use a temporal split; shuffle=False.",
            "Correlations are exploratory instruments, not trading proof.",
            "Cost-floor fields are descriptive screens; they do not prove executable edge.",
        ],
    }
    assert_valid_contract(result, "eso.asset_dissect.v1")
    return {
        "result": result,
        "correlations": corr_df,
        "quantiles": quantile_df,
        "phase_sectors": sector_df,
        "targets": targets,
    }


def write_asset_dissect_bundle(dissected: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(dissected["result"])
    paths = {
        "correlations_csv": output_dir / "feature_target_correlations.csv",
        "quantiles_csv": output_dir / "quantile_regimes.csv",
        "phase_sectors_csv": output_dir / "phase_sectors.csv",
        "targets_csv": output_dir / "targets.csv",
    }
    dissected["correlations"].to_csv(paths["correlations_csv"], index=False)
    dissected["quantiles"].to_csv(paths["quantiles_csv"], index=False)
    dissected["phase_sectors"].to_csv(paths["phase_sectors_csv"], index=False)
    dissected["targets"].to_csv(paths["targets_csv"], index=False)
    result["tables"] = {name: str(path) for name, path in paths.items()}
    assert_valid_contract(result, "eso.asset_dissect.v1")
    json_path = output_dir / "asset_dissect.json"
    json_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"asset_dissect_json": str(json_path), **{k: str(v) for k, v in paths.items()}}
