"""Tests for deployment helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eso.lab.deployment import (
    TradeAction,
    compute_trade_diff,
    write_trade_actions,
)


def _make_live_signal(tmp_path) -> str:
    payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "target_annual_vol": 0.30,
        "per_asset": [
            {"asset": "AAA", "selection_sharpe": 2.0, "current_position": 1.0,
             "selected_signal": "cos_theta_24h"},
            {"asset": "BBB", "selection_sharpe": -0.1, "current_position": 1.0,
             "selected_signal": "cos_theta_24h"},
            {"asset": "CCC", "selection_sharpe": 1.0, "current_position": 0.0,
             "selected_signal": "cos_theta_24h"},
        ],
        "vol_scalers": {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
        "portfolio_positions": {"AAA": 0.20, "BBB": 0.20, "CCC": 0.00},
        "notes": [],
    }
    path = tmp_path / "live.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_compute_trade_diff_basic(tmp_path):
    path = _make_live_signal(tmp_path)
    actions = compute_trade_diff(path)
    by_asset = {a.asset: a for a in actions}
    assert by_asset["AAA"].action == "BUY"
    # BBB has selection_sharpe -0.1 → fails quality gate → HOLD
    assert by_asset["BBB"].action == "HOLD"
    assert "selection_sharpe" in by_asset["BBB"].notes
    # CCC target is 0 with held=0 → HOLD by min_trade_size
    assert by_asset["CCC"].action == "HOLD"


def test_compute_trade_diff_with_held(tmp_path):
    path = _make_live_signal(tmp_path)
    held = {"AAA": 0.20, "BBB": 0.0, "CCC": 0.0}
    actions = compute_trade_diff(path, held_positions=held)
    by_asset = {a.asset: a for a in actions}
    # AAA already at target → HOLD
    assert by_asset["AAA"].action == "HOLD"


def test_compute_trade_diff_respects_min_trade(tmp_path):
    path = _make_live_signal(tmp_path)
    held = {"AAA": 0.18}  # delta = 0.02, below default 0.05
    actions = compute_trade_diff(path, held_positions=held)
    by_asset = {a.asset: a for a in actions}
    assert by_asset["AAA"].action == "HOLD"


def test_funding_merge_fills_nan_rows(tmp_path, monkeypatch):
    """Regression: funding refresh must actually fill NaN funding_rate values.

    The original implementation used merge_asof with suffixes=("", "_new")
    and then compared `merged["funding_rate"]` (the OLD column) for the
    update mask — which is always NaN where the existing CSV's funding is
    NaN, so no updates ever fired.
    """
    import pandas as pd
    from eso.lab import deployment as dep

    # Build a CSV with 5 bars; first 3 have funding, last 2 are NaN
    csv = tmp_path / "TST_4h.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC"),
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5],
        "volume": [10, 20, 30, 40, 50],
        "funding_rate": [0.0001, 0.0001, 0.0001, None, None],
    }).to_csv(csv, index=False)

    # Stub fetch to return a funding observation that pre-dates the NaN bars
    def stub_fetch(symbol, limit=200, category="linear"):
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-01-01 08:00:00+00:00",  # before NaN bars
                "2025-01-01 12:00:00+00:00",  # covers the first NaN bar
            ]),
            "funding_rate": [0.0002, 0.0003],
        })
    monkeypatch.setattr(dep, "fetch_bybit_funding", stub_fetch)

    n = dep.refresh_funding_in_csv("TST", csv)
    assert n == 2, f"Expected 2 NaN rows filled, got {n}"

    after = pd.read_csv(csv)
    assert after["funding_rate"].isna().sum() == 0
    # Last NaN bar should pick up the most recent prior funding (0.0003)
    assert after["funding_rate"].iloc[-1] == pytest.approx(0.0003)


def test_write_trade_actions(tmp_path):
    actions = [
        TradeAction("AAA", 0.0, 0.2, 0.2, "BUY", True, ""),
        TradeAction("BBB", 0.1, 0.1, 0.0, "HOLD", True, ""),
    ]
    out = tmp_path / "actions.json"
    write_trade_actions(actions, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["actions"]) == 2
    assert payload["actions"][0]["action"] == "BUY"
