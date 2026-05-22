"""Vectorized cost-aware backtester.

Honest accounting rules:
    - Position at bar t is held from close[t] to close[t+1].
    - Returns at t+1 use log(close[t+1] / close[t]).
    - Fees and slippage are charged on |position[t] - position[t-1]| (turnover).
    - No lookahead: any signal generating position[t] must use only data up to t.

Default costs target Bybit USDT perpetuals (taker), which is the realistic
cost when crossing the book. Maker rebates are not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# Bybit perpetual taker fee = 0.055% (5.5 bps). Round-trip = 11 bps.
# Conservative slippage on liquid pairs at modest size = 2 bps per side.
DEFAULT_TAKER_FEE_BPS = 5.5
DEFAULT_SLIPPAGE_BPS = 2.0


@dataclass
class BacktestConfig:
    fee_bps: float = DEFAULT_TAKER_FEE_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    bars_per_year: int = 24 * 365  # 1h bars
    initial_equity: float = 1.0
    allow_short: bool = True

    @property
    def cost_per_unit_turnover(self) -> float:
        """Fraction lost per unit of |Δposition| (one leg)."""
        return (self.fee_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Period:        {m['n_bars']} bars  ({m['years']:.2f} years)",
            f"Trades:        {m['n_trades']}  (turnover/yr={m['turnover_per_year']:.1f})",
            f"Gross return:  {m['gross_total_return']*100:+.2f}%   "
            f"Sharpe={m['gross_sharpe']:.2f}",
            f"Net return:    {m['net_total_return']*100:+.2f}%   "
            f"Sharpe={m['net_sharpe']:.2f}",
            f"Cost drag:     {m['cost_drag']*100:.2f}%   "
            f"({m['cost_drag_per_year']*100:.2f}%/yr)",
            f"Max drawdown:  {m['max_drawdown']*100:.2f}%",
            f"Hit rate:      {m['hit_rate']*100:.1f}%   "
            f"(of {m['active_bars']} active bars)",
            f"Verdict:       {m['verdict']}",
        ]
        return "\n".join(lines)


def _compute_metrics(
    gross: np.ndarray,
    net: np.ndarray,
    positions: np.ndarray,
    turnover: np.ndarray,
    cfg: BacktestConfig,
) -> dict:
    n = len(net)
    years = n / cfg.bars_per_year if cfg.bars_per_year > 0 else 0.0

    def sharpe(r: np.ndarray) -> float:
        sd = r.std(ddof=0)
        if sd <= 0 or len(r) == 0:
            return 0.0
        return float(r.mean() / sd * np.sqrt(cfg.bars_per_year))

    equity_gross = np.exp(np.cumsum(gross))
    equity_net = np.exp(np.cumsum(net))
    peak = np.maximum.accumulate(equity_net)
    dd = (equity_net / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Position changes = trades. Each non-zero turnover bar = 1 trade.
    n_trades = int((turnover > 0).sum())
    turnover_per_year = (turnover.sum() / years) if years > 0 else 0.0

    active_mask = positions != 0
    active_bars = int(active_mask.sum())
    if active_bars > 0:
        # Hit rate: fraction of active bars where sign(pos) * gross_return > 0
        signed = np.sign(positions[active_mask]) * gross[active_mask]
        hit_rate = float((signed > 0).mean())
    else:
        hit_rate = 0.0

    gross_total = float(np.expm1(gross.sum()))
    net_total = float(np.expm1(net.sum()))
    cost_drag = gross_total - net_total
    cost_drag_per_year = cost_drag / years if years > 0 else 0.0

    net_sharpe = sharpe(net)
    # Heuristic verdict — low bar deliberately. Real production needs more.
    if net_sharpe >= 1.0 and net_total > 0:
        verdict = "PROMISING — net Sharpe >= 1.0 after costs"
    elif net_sharpe >= 0.5 and net_total > 0:
        verdict = "MARGINAL — signal exists but costs eat most edge"
    elif net_total > 0:
        verdict = "WEAK — barely positive, not tradeable as-is"
    else:
        verdict = "NEGATIVE — signal does not survive costs"

    return {
        "n_bars": n,
        "years": years,
        "n_trades": n_trades,
        "turnover_per_year": float(turnover_per_year),
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "cost_drag": cost_drag,
        "cost_drag_per_year": cost_drag_per_year,
        "gross_sharpe": sharpe(gross),
        "net_sharpe": net_sharpe,
        "max_drawdown": max_dd,
        "active_bars": active_bars,
        "hit_rate": hit_rate,
        "verdict": verdict,
    }


def run_backtest(
    prices: pd.Series,
    positions: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run a cost-aware backtest.

    Args:
        prices:    Close prices indexed by timestamp (or bar index).
        positions: Position per bar in [-1, 1]. Held from current close to next.
                   Must share index with prices. NaN treated as 0.
        config:    Cost/calendar configuration.

    Returns:
        BacktestResult with equity curve, returns, turnover and metrics.
    """
    cfg = config or BacktestConfig()

    if len(prices) != len(positions):
        raise ValueError(
            f"prices ({len(prices)}) and positions ({len(positions)}) "
            "must have the same length"
        )
    if len(prices) < 2:
        raise ValueError("Need at least 2 bars to backtest.")

    px = prices.astype(float).to_numpy()
    pos = positions.astype(float).fillna(0.0).to_numpy()

    if not cfg.allow_short:
        pos = np.clip(pos, 0.0, 1.0)
    else:
        pos = np.clip(pos, -1.0, 1.0)

    # Log returns aligned: ret[t] applies to bar t -> t+1
    log_ret = np.diff(np.log(px))  # len n-1
    pos_held = pos[:-1]              # position held during that bar

    gross = pos_held * log_ret

    # Turnover at the point of taking new position. We assume the trader sets
    # position[t] near close[t] and pays cost based on |pos[t] - pos[t-1]|.
    # First bar: initial entry from 0 to pos[0].
    delta = np.empty_like(pos)
    delta[0] = np.abs(pos[0])
    delta[1:] = np.abs(np.diff(pos))
    # Align turnover cost to the same time axis as gross returns:
    # cost incurred at bar t reduces the return earned during bar t -> t+1.
    turnover_aligned = delta[:-1]  # cost charged before the bar's return is earned
    cost = turnover_aligned * cfg.cost_per_unit_turnover

    # Net log return ≈ gross - cost (cost is small enough that log≈linear).
    net = gross - cost

    idx = prices.index[1:]
    equity = pd.Series(cfg.initial_equity * np.exp(np.cumsum(net)), index=idx, name="equity")
    gross_s = pd.Series(gross, index=idx, name="gross_log_return")
    net_s = pd.Series(net, index=idx, name="net_log_return")
    pos_s = pd.Series(pos_held, index=idx, name="position")
    turn_s = pd.Series(turnover_aligned, index=idx, name="turnover")

    metrics = _compute_metrics(gross, net, pos_held, turnover_aligned, cfg)
    metrics["config"] = asdict(cfg)

    return BacktestResult(
        equity_curve=equity,
        gross_returns=gross_s,
        net_returns=net_s,
        positions=pos_s,
        turnover=turn_s,
        metrics=metrics,
    )
