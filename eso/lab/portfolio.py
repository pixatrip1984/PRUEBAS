"""Equal-weight portfolio combination of walk-forward results.

Aligns net-return streams across assets by timestamp, equal-weights them,
and reports the portfolio aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import json
import numpy as np
import pandas as pd

from eso.lab.walk_forward import WalkForwardResult, walk_forward_backtest


@dataclass
class PortfolioReport:
    assets: list
    n_bars_union: int
    n_bars_intersection: int
    portfolio_net_return: float
    portfolio_net_sharpe: float
    portfolio_max_drawdown: float
    avg_assets_active: float
    annualisation: int


def _build_returns_frame(results: list[WalkForwardResult]) -> pd.DataFrame:
    cols = {}
    for r in results:
        if not r.timestamps or not r.net_returns:
            continue
        s = pd.Series(r.net_returns, index=pd.to_datetime(r.timestamps), name=r.asset)
        cols[r.asset] = s[~s.index.duplicated(keep="first")]
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def equal_weight_portfolio(
    results: list[WalkForwardResult],
    bars_per_year: int = 2190,
    output_dir: str | Path | None = None,
) -> PortfolioReport:
    """Equal-weight portfolio across walk-forward equity curves.

    For each bar where at least one asset has a position, that asset's
    contribution is 1/n_active. Bars where no asset is active contribute zero.
    """
    df = _build_returns_frame(results)
    if df.empty:
        raise ValueError("No walk-forward results with timestamped returns.")

    union_idx = df.index
    intersection_mask = df.notna().all(axis=1)

    # At each bar, equal-weight across assets that actually have a return value
    active_mask = df.notna()
    n_active = active_mask.sum(axis=1).replace(0, np.nan)
    avg_active = float(n_active.mean())
    weighted = df.fillna(0.0).div(n_active.fillna(1), axis=0)
    port_ret = weighted.sum(axis=1)  # log returns

    cum_log = port_ret.cumsum()
    equity = np.exp(cum_log)
    peak = equity.cummax()
    dd = equity / peak - 1.0

    sd = float(port_ret.std(ddof=0))
    sharpe = (
        float(port_ret.mean() / sd * np.sqrt(bars_per_year))
        if sd > 0 else 0.0
    )

    report = PortfolioReport(
        assets=[r.asset for r in results if r.net_returns],
        n_bars_union=int(active_mask.any(axis=1).sum()),
        n_bars_intersection=int(intersection_mask.sum()),
        portfolio_net_return=float(np.expm1(cum_log.iloc[-1])),
        portfolio_net_sharpe=sharpe,
        portfolio_max_drawdown=float(dd.min()),
        avg_assets_active=avg_active,
        annualisation=bars_per_year,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "portfolio_report.json").write_text(
            json.dumps(asdict(report), indent=2, default=float),
            encoding="utf-8",
        )
        df_curve = pd.DataFrame({
            "portfolio_log_return": port_ret,
            "equity": equity,
            "drawdown": dd,
            "n_active": n_active,
        })
        df_curve.to_csv(out / "portfolio_curve.csv")
        # Markdown
        md = [
            "# Equal-weight portfolio — walk-forward",
            "",
            f"Assets: {', '.join(report.assets)}",
            f"Bars (union): {report.n_bars_union}",
            f"Bars (all assets active): {report.n_bars_intersection}",
            f"Avg assets active per bar: {report.avg_assets_active:.2f}",
            "",
            f"**Portfolio net return:** {report.portfolio_net_return*100:+.1f}%",
            f"**Portfolio net Sharpe:** {report.portfolio_net_sharpe:+.2f}",
            f"**Max drawdown:** {report.portfolio_max_drawdown*100:+.1f}%",
        ]
        (out / "portfolio_report.md").write_text("\n".join(md), encoding="utf-8")

    return report


def run_portfolio(
    csv_paths: list[str | Path],
    output_dir: str | Path = "reports/portfolio_walkforward",
    n_folds: int = 4,
    bars_per_year: int = 2190,
) -> PortfolioReport:
    results = []
    for p in csv_paths:
        try:
            r = walk_forward_backtest(p, n_folds=n_folds, bars_per_year=bars_per_year)
        except Exception as exc:
            print(f"  {Path(p).stem} failed: {exc}")
            continue
        results.append(r)
    return equal_weight_portfolio(results, bars_per_year=bars_per_year,
                                  output_dir=output_dir)
