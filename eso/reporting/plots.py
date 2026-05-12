"""Plot generation for ESO reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eso.diagnostics.stationarity import recurrence_plot
from eso.diagnostics.symmetries import autocorrelation
from eso.topology.reconstruction import project_data_to_manifold
from eso.topology.manifolds import get_manifold


def _series(data):
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        return x
    return x[:, 0]


def save_time_series(data, outdir: Path) -> str:
    y = _series(data)
    path = outdir / "time_series_overview.png"
    plt.figure(figsize=(10, 3))
    plt.plot(y[: min(len(y), 2000)])
    plt.title("Time series overview (first numeric feature)")
    plt.xlabel("index")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def save_recurrence(data, outdir: Path) -> str:
    path = outdir / "recurrence_plot.png"
    rp = recurrence_plot(data, max_points=400)
    plt.figure(figsize=(5, 5))
    plt.imshow(rp, aspect="auto", origin="lower")
    plt.title("Recurrence plot")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def save_fft(data, outdir: Path) -> str:
    y = _series(data)
    y = y - np.mean(y)
    freqs = np.fft.rfftfreq(len(y))
    power = np.abs(np.fft.rfft(y)) ** 2
    path = outdir / "fft_spectrum.png"
    plt.figure(figsize=(8, 3))
    plt.plot(freqs[1:], power[1:])
    plt.title("FFT spectrum")
    plt.xlabel("frequency")
    plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def save_autocorrelation(data, outdir: Path) -> str:
    corr = autocorrelation(data, max_lag=min(512, max(2, len(_series(data)) // 2)))
    path = outdir / "autocorrelation.png"
    plt.figure(figsize=(8, 3))
    plt.plot(corr)
    plt.title("Autocorrelation")
    plt.xlabel("lag")
    plt.ylabel("corr")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def save_ranking(evaluations: list[dict], outdir: Path) -> str:
    labels = [e["manifold"] for e in evaluations]
    values = [e.get("score_mean", e.get("score", np.nan)) for e in evaluations]
    path = outdir / "manifold_ranking.png"
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Manifold ranking score")
    plt.ylabel("score (lower is better)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def save_latent_projection(data, manifold_name: str, outdir: Path) -> str:
    mani = get_manifold(manifold_name)
    z = project_data_to_manifold(data, mani)
    path = outdir / f"latent_projection_{mani.name}.png"
    c = _series(data)
    plt.figure(figsize=(5, 4))
    if z.shape[1] >= 3:
        ax = plt.axes(projection="3d")
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, s=8)
        ax.set_title(f"Latent projection: {mani.name}")
    else:
        plt.scatter(z[:, 0], z[:, 1], c=c, s=8)
        plt.title(f"Latent projection: {mani.name}")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def generate_figures(data, report: dict, outdir: str | Path) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = {
        "time_series_overview": save_time_series(data, outdir),
        "recurrence_plot": save_recurrence(data, outdir),
        "fft_spectrum": save_fft(data, outdir),
        "autocorrelation": save_autocorrelation(data, outdir),
        "manifold_ranking": save_ranking(report.get("evaluations", []), outdir),
    }
    for e in report.get("evaluations", [])[:4]:
        key = f"latent_projection_{e['manifold']}"
        figures[key] = save_latent_projection(data, e["manifold"], outdir)
    return figures
