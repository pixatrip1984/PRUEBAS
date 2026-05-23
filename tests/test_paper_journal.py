"""Smoke tests for paper-trading journal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eso.lab.paper_journal import simulate_paper_trading


def _make_csv(tmp_path, n: int = 1500, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    rets = 0.005 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.001, "low": close * 0.999,
        "close": close,
        "volume": rng.lognormal(10, 0.3, n),
    })
    p = tmp_path / "TEST_4h.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_simulate_paper_trading_produces_entries(tmp_path):
    csv = _make_csv(tmp_path, n=1200)
    entries, summary = simulate_paper_trading(
        csv, start_bar_offset=-50, output_dir=str(tmp_path / "out"),
    )
    assert summary.n_steps > 0
    assert len(entries) == summary.n_steps
    assert (tmp_path / "out" / "journal_TEST_4h.csv").exists()


def test_paper_trading_held_position_evolves(tmp_path):
    csv = _make_csv(tmp_path, n=1200, seed=7)
    entries, _ = simulate_paper_trading(csv, start_bar_offset=-50)
    # At least one held position should differ from zero somewhere
    held_values = [abs(e.held_position) for e in entries]
    assert max(held_values) > 0 or all(e.target_position == 0 for e in entries)
