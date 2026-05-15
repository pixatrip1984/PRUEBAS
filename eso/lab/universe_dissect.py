"""Run asset dissection over a declarative asset universe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eso.contracts import assert_valid_contract
from eso.data.features import prepare_btc_klines
from eso.data.loader import read_table
from eso.lab.asset_dissect import dissect_feature_frame, write_asset_dissect_bundle
from eso.lab.context import append_reference_context
from eso.modeling import build_feature_frame


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        value = obj.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_") or "asset"


def _slice_rows(df: pd.DataFrame, skip_rows: int | None, max_rows: int | None) -> pd.DataFrame:
    if skip_rows:
        df = df.iloc[int(skip_rows):]
    if max_rows:
        df = df.head(int(max_rows))
    return df.reset_index(drop=True)


def _load_universe(universe: str | Path | dict) -> dict:
    if isinstance(universe, dict):
        payload = universe
    else:
        payload = json.loads(Path(universe).read_text(encoding="utf-8"))
    assert_valid_contract(payload, "eso.asset_universe.v1")
    return payload


def _merge_asset_config(defaults: dict, asset: dict) -> dict:
    merged = dict(defaults)
    merged.update(asset)
    return merged


def _best_feature(row_df: pd.DataFrame, col: str) -> tuple[str | None, float | None]:
    if row_df.empty or col not in row_df.columns:
        return None, None
    values = row_df[["feature", col]].dropna()
    if values.empty:
        return None, None
    idx = values[col].abs().idxmax()
    row = values.loc[idx]
    return str(row["feature"]), float(row[col])


def _asset_aggregate(asset_id: str, dissected: dict) -> list[dict]:
    corr = dissected["correlations"]
    target_summary = {
        int(item["horizon"]): item for item in dissected["result"].get("target_summary", [])
    }
    rows = []
    for horizon, grp in corr.groupby("horizon"):
        rv_feature, rv_corr = _best_feature(grp, "corr_rv_test")
        lr_feature, lr_corr = _best_feature(grp, "corr_lr_test")
        dir_feature, dir_corr = _best_feature(grp, "corr_direction_test")
        ts = target_summary.get(int(horizon), {})
        rows.append({
            "asset_id": asset_id,
            "horizon": int(horizon),
            "rows": dissected["result"].get("rows"),
            "train_size": dissected["result"].get("split", {}).get("train_size"),
            "test_size": dissected["result"].get("split", {}).get("test_size"),
            "direction_baseline_accuracy": ts.get("direction_baseline_accuracy"),
            "test_direction_baseline_accuracy": ts.get("test_direction_baseline_accuracy"),
            "future_rv_median": ts.get("future_rv_median"),
            "top_rv_feature": rv_feature,
            "top_rv_corr_test": rv_corr,
            "top_lr_feature": lr_feature,
            "top_lr_corr_test": lr_corr,
            "top_direction_feature": dir_feature,
            "top_direction_corr_test": dir_corr,
        })
    return rows


def run_universe_dissect(
    universe: str | Path | dict,
    output_dir: str | Path,
) -> dict:
    """Run asset dissection for every asset in a universe spec."""
    spec = _load_universe(universe)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = spec.get("defaults", {})
    aggregate_rows: list[dict] = []
    assets = []
    errors = []

    for asset in spec.get("assets", []):
        cfg = _merge_asset_config(defaults, asset)
        asset_id = cfg.get("asset_id") or Path(cfg["path"]).stem
        asset_dir = output_dir / _safe_name(asset_id)
        try:
            raw = read_table(cfg["path"])
            if cfg.get("klines_format", False):
                raw = prepare_btc_klines(raw)
            raw = _slice_rows(raw, cfg.get("skip_rows"), cfg.get("max_rows"))
            fv = build_feature_frame(
                raw,
                feature_mode=cfg.get("feature_mode", "compact"),
                columns=cfg.get("columns"),
                close_col=cfg.get("close_col", "close"),
                umap_train_size=cfg.get("umap_train_size", 27000),
                smooth_windows=tuple(cfg.get("smooth_windows", [6, 24, 72])),
            )
            reference = None
            if cfg.get("reference_path"):
                reference = read_table(cfg["reference_path"])
                if cfg.get("reference_klines_format", False):
                    reference = prepare_btc_klines(reference)
                fv = append_reference_context(
                    fv,
                    raw,
                    reference,
                    timestamp_col=cfg.get("timestamp_col", "timestamp"),
                    close_col=cfg.get("close_col", "close"),
                    windows=tuple(cfg.get("relative_windows", [24, 72])),
                    tolerance=cfg.get("reference_tolerance"),
                )
            dissected = dissect_feature_frame(
                fv,
                asset_id=asset_id,
                horizons=tuple(cfg.get("horizons", [3, 12])),
                train_size=cfg.get("train_size"),
                train_fraction=cfg.get("train_fraction", 0.7),
                phase_window=cfg.get("phase_window", 24),
            )
            artifacts = write_asset_dissect_bundle(dissected, asset_dir)
            aggregate_rows.extend(_asset_aggregate(asset_id, dissected))
            assets.append({
                "asset_id": asset_id,
                "path": cfg["path"],
                "rows": dissected["result"]["rows"],
                "artifacts": artifacts,
            })
        except Exception as exc:
            errors.append({"asset_id": asset_id, "path": cfg.get("path"), "error": str(exc)})

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = output_dir / "asset_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    leaders = aggregate.copy()
    if not leaders.empty:
        leaders["abs_top_rv_corr_test"] = leaders["top_rv_corr_test"].abs()
        leaders["abs_top_direction_corr_test"] = leaders["top_direction_corr_test"].abs()
        leaders = leaders.sort_values(
            ["horizon", "abs_top_rv_corr_test", "abs_top_direction_corr_test"],
            ascending=[True, False, False],
        )
    leaders_path = output_dir / "asset_leaders.csv"
    leaders.to_csv(leaders_path, index=False)

    result = {
        "schema_version": "eso.universe_dissect.v1",
        "name": spec.get("name"),
        "output_dir": str(output_dir),
        "asset_count": len(assets),
        "error_count": len(errors),
        "assets": assets,
        "errors": errors,
        "tables": {
            "asset_summary_csv": str(aggregate_path),
            "asset_leaders_csv": str(leaders_path),
        },
        "guardrails": [
            "Each asset uses the same dissection engine as single-asset dissect.",
            "Aggregate correlations are exploratory and must be validated by model_eval/benchmark before strategy use.",
        ],
    }
    assert_valid_contract(result, "eso.universe_dissect.v1")
    return result


def write_universe_dissect_result(result: dict, output_path: str | Path) -> str:
    assert_valid_contract(result, "eso.universe_dissect.v1")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
