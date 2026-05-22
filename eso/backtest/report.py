"""Report writer for backtest runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eso.backtest.engine import BacktestResult


def write_backtest_report(
    result: BacktestResult,
    output_dir: str | Path,
    title: str = "ESO Backtest",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, default=float),
        encoding="utf-8",
    )

    # Equity curve CSV (compact)
    curve = result.equity_curve.to_frame()
    curve["position"] = result.positions.values
    curve["net_log_return"] = result.net_returns.values
    curve_path = out / "equity_curve.csv"
    curve.to_csv(curve_path, index=True)

    # PnL figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            label="Net equity (after costs)", color="C0", linewidth=1.2)
    gross_equity = (result.gross_returns.cumsum().apply("exp"))
    ax.plot(gross_equity.index, gross_equity.values,
            label="Gross equity", color="C1", alpha=0.6, linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Equity (1.0 = initial)")
    ax.set_title(f"{title} — net Sharpe {result.metrics['net_sharpe']:.2f}, "
                 f"net total {result.metrics['net_total_return']*100:+.1f}%")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(result.positions.index, result.positions.values,
             color="C2", linewidth=0.6)
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out / "equity.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    # Markdown summary
    md = [
        f"# {title}",
        "",
        "```",
        result.summary(),
        "```",
        "",
        f"- Equity curve: `{curve_path.name}`",
        f"- Figure: `{fig_path.name}`",
        f"- Metrics: `{metrics_path.name}`",
    ]
    md_path = out / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    return {
        "metrics_json": str(metrics_path),
        "equity_csv": str(curve_path),
        "figure": str(fig_path),
        "summary_md": str(md_path),
    }
