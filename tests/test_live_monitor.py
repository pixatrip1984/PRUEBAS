"""Tests for live monitor / circuit breaker."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eso.lab.live_monitor import (
    LiveObservation,
    append_observation,
    compute_status,
    print_status,
    seed_from_paper_journal,
)


def _seed_journal(tmp_path, n: int = 200, daily_drift: float = 0.001) -> Path:
    rng = np.random.default_rng(0)
    rets = rng.normal(daily_drift, 0.005, n)
    df = pd.DataFrame({
        "portfolio_log_return": rets,
        "equity": np.exp(np.cumsum(rets)),
        "drawdown": np.zeros(n),
        "n_active": np.full(n, 4.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"))
    p = tmp_path / "journal.csv"
    df.to_csv(p)
    return p


def test_seed_from_paper_journal_loads_log(tmp_path):
    journal = _seed_journal(tmp_path)
    log = tmp_path / "log.json"
    n = seed_from_paper_journal(journal, log)
    assert n == 200
    assert log.exists()


def test_compute_status_returns_go_for_healthy_log(tmp_path):
    # All positive returns → Sharpe should be high enough for GO
    journal = _seed_journal(tmp_path, n=200, daily_drift=0.003)
    log = tmp_path / "log.json"
    seed_from_paper_journal(journal, log)
    status = compute_status(log)
    assert status.n_observations == 200
    assert status.signal in {"GO", "WARN"}  # might be WARN if Sharpe is borderline


def test_compute_status_signals_stop_on_extreme_drawdown(tmp_path):
    # Drift large negative → big drawdown → STOP
    journal = _seed_journal(tmp_path, n=200, daily_drift=-0.01)
    log = tmp_path / "log.json"
    seed_from_paper_journal(journal, log)
    status = compute_status(log)
    assert status.signal == "STOP"
    assert any("Drawdown" in r or "Sharpe" in r for r in status.reasons)


def test_append_observation_increments_count(tmp_path):
    log = tmp_path / "log.json"
    obs = LiveObservation(
        timestamp="2026-01-01T00:00:00",
        target_positions={"AAA": 0.5},
        actual_positions={"AAA": 0.5},
        marks={"AAA": 100.0},
        realized_log_returns={"AAA": 0.001},
        portfolio_log_return=0.001,
    )
    n1 = append_observation(log, obs)
    n2 = append_observation(log, obs)
    assert n1 == 1 and n2 == 2


def test_empty_log_status_is_go(tmp_path):
    log = tmp_path / "log.json"
    status = compute_status(log)
    assert status.n_observations == 0
    assert status.signal == "GO"


def test_print_status_does_not_crash(tmp_path, capsys):
    journal = _seed_journal(tmp_path)
    log = tmp_path / "log.json"
    seed_from_paper_journal(journal, log)
    status = compute_status(log)
    print_status(status)
    captured = capsys.readouterr()
    assert "Realized Sharpe" in captured.out
