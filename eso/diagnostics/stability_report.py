"""Plotting and report writer for stability analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eso.diagnostics.signal_stability import StabilityReport


def write_stability_report(
    report: StabilityReport,
    output_dir: str | Path,
    title: str = "Signal stability",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "target_horizon": report.target_horizon,
        "window": report.window,
        "mean_corr": report.mean_corr,
        "flip_count": report.flip_count,
        "cross_corr_with_regime": report.cross_corr_with_regime,
    }
    (out / "stability.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    # Markdown
    (out / "stability.md").write_text(
        f"# {title}\n\n```\n{report.summary}\n```\n",
        encoding="utf-8",
    )

    # Figure: rolling correlation per feature + regime overlay
    rc = report.rolling_corr
    ri = report.regime_indicators

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    for col in rc.columns:
        ax.plot(rc.index, rc[col].values, label=col, linewidth=0.8, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_ylabel(f"Rolling corr (window={report.window})\nvs +{report.target_horizon}h return")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if not ri.empty:
        for col in ri.columns:
            v = ri[col].values
            # Min-max normalise each indicator for readability on shared axis
            vmin, vmax = float(min(v[~(v != v).astype(bool)] if v.size else [0])), \
                         float(max(v[~(v != v).astype(bool)] if v.size else [1]))
            # Robust normalize via percentiles
            import numpy as np
            valid = v[~np.isnan(v)]
            if len(valid) > 0:
                p5, p95 = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
                rng = (p95 - p5) or 1.0
                norm = (v - p5) / rng
                ax2.plot(ri.index, norm, label=col, linewidth=0.7, alpha=0.85)
    ax2.set_ylabel("Regime (norm)")
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "stability.png", dpi=110)
    plt.close(fig)

    return {
        "json": str(out / "stability.json"),
        "md": str(out / "stability.md"),
        "figure": str(out / "stability.png"),
    }
