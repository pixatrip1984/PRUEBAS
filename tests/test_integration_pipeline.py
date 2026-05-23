"""End-to-end integration test for the complete deployment pipeline.

Exercises in sequence:
    1. Synthetic OHLCV + funding CSV creation
    2. build_alt_feature_vector with include_funding
    3. gated_phase_threshold strategy + backtest
    4. walk_forward_backtest
    5. asset_live_signal
    6. compute_trade_diff with quality gate
    7. simulate_paper_trading
    8. live monitor seed + compute_status

If any step's API changes incompatibly, this test catches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_csv(tmp_path: Path, n: int = 1600, with_funding: bool = False,
                   filename: str = "INTEG_4h.csv") -> Path:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    rets = 0.004 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.lognormal(10, 0.3, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })
    if with_funding:
        df["funding_rate"] = rng.normal(0.0001, 0.0002, n)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def test_pipeline_runs_end_to_end(tmp_path):
    """Run every stage of the deployment pipeline sequentially.

    No assertions on specific numbers — only that each stage produces
    a non-empty result with the expected schema.
    """
    csv = _synthetic_csv(tmp_path, n=1600, with_funding=False)

    # Stage 1: feature vector
    from eso.lab.multi_asset_sweep import build_alt_feature_vector
    fv = build_alt_feature_vector(pd.read_csv(csv), include_funding=False)
    assert len(fv) > 100
    for col in ["cos_theta_24h", "ring_radius", "close"]:
        assert col in fv.columns

    # Stage 2: strategy + backtest
    from eso.backtest import (
        BacktestConfig,
        gated_phase_threshold_strategy,
        run_backtest,
    )
    pos = gated_phase_threshold_strategy(
        fv, signal_col="cos_theta_24h", gate_col="ring_radius",
        gate_percentile=0.50, gate_window=200,
        enter_threshold=0.3, exit_threshold=0.1,
    )
    cfg = BacktestConfig(bars_per_year=2190)
    bt = run_backtest(fv["close"], pos, cfg)
    assert "net_sharpe" in bt.metrics

    # Stage 3: walk-forward
    from eso.lab.walk_forward import walk_forward_backtest
    wf = walk_forward_backtest(str(csv), n_folds=4, bars_per_year=2190)
    assert wf.aggregate_trades >= 0
    assert len(wf.folds) <= 3

    # Stage 4: live signal
    from eso.lab.live_signal import asset_live_signal
    sig = asset_live_signal(str(csv), selection_window_bars=200,
                            bars_per_year=2190)
    assert sig is not None
    assert -1.0 <= sig.current_position <= 1.0

    # Stage 5: trade diff
    from eso.lab.live_signal import get_live_positions
    out = tmp_path / "live.json"
    rep = get_live_positions(
        configs=[{"csv": str(csv), "include_funding": False}],
        output_path=str(out),
        selection_window_bars=200,
    )
    assert out.exists()

    from eso.lab.deployment import compute_trade_diff
    actions = compute_trade_diff(out, held_positions={},
                                  min_trade_size=0.05, min_selection_sharpe=0.5)
    assert len(actions) == 1
    assert actions[0].action in {"BUY", "SELL", "HOLD"}

    # Stage 6: paper journal
    from eso.lab.paper_journal import simulate_paper_trading
    entries, summary = simulate_paper_trading(
        str(csv), start_bar_offset=-30,
        selection_window_bars=200,
        output_dir=str(tmp_path / "paper"),
    )
    assert summary.n_steps > 0

    # Stage 7: live monitor
    journal_csv = tmp_path / "paper_for_monitor.csv"
    pd.DataFrame({
        "portfolio_log_return": [e.realized_log_return for e in entries],
        "equity": np.exp(np.cumsum([e.realized_log_return for e in entries])),
        "drawdown": np.zeros(len(entries)),
        "n_active": np.ones(len(entries)),
    }, index=pd.to_datetime([e.timestamp for e in entries])).to_csv(journal_csv)

    from eso.lab.live_monitor import seed_from_paper_journal, compute_status
    log_path = tmp_path / "monitor.json"
    n_seeded = seed_from_paper_journal(journal_csv, log_path)
    assert n_seeded == len(entries)

    status = compute_status(log_path)
    assert status.signal in {"GO", "WARN", "STOP"}
    assert status.n_observations == n_seeded
