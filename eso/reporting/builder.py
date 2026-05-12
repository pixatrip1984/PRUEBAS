"""Build complete ESO report bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .plots import generate_figures


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


def agent_summary(report: dict) -> dict:
    best = report.get("best") or {}
    diagnosis = report.get("diagnosis", {})
    dim = diagnosis.get("dimension", {}).get("consensus_dimension")
    manifold = best.get("manifold", "unknown")
    err = best.get("reconstruction_error_mean", best.get("reconstruction_error"))
    warnings = []
    if best.get("latent_utilization", 1.0) < 0.1:
        warnings.append("low latent utilization")
    if err is None:
        confidence = "low"
    elif best.get("reconstruction_error_std", 0.0) > max(abs(err), 1e-12):
        confidence = "low"
    else:
        confidence = "medium"
    return {
        "one_sentence": f"Best current candidate is {manifold} with intrinsic dimension estimate {dim}.",
        "recommended_next_action": "Inspect report.html and run a second explore pass with more rows/masks if confidence is not high.",
        "confidence": confidence,
        "warnings": warnings,
    }


def write_metrics(report: dict, output_dir: Path) -> str:
    rows = []
    for e in report.get("evaluations", []):
        row = {k: v for k, v in e.items() if k != "validation_runs"}
        rows.append(row)
    path = output_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def write_markdown(report: dict, figures: dict, output_dir: Path) -> str:
    best = report.get("best") or {}
    diagnosis = report.get("diagnosis", {})
    lines = []
    lines.append(f"# ESO Report — {report.get('dataset_id', 'dataset')}")
    lines.append("")
    lines.append("## 1. Resumen ejecutivo")
    lines.append(f"- Mejor geometría candidata: **{best.get('manifold', 'unknown')}**")
    lines.append(f"- Score: `{best.get('score_mean', best.get('score', 'n/a'))}`")
    lines.append(f"- Error reconstrucción: `{best.get('reconstruction_error_mean', best.get('reconstruction_error', 'n/a'))}`")
    lines.append(f"- Dimensión intrínseca estimada: `{diagnosis.get('dimension', {}).get('consensus_dimension', 'n/a')}`")
    lines.append(f"- Resumen diagnóstico: {diagnosis.get('summary', 'n/a')}")
    lines.append("")
    lines.append("## 2. Datos de entrada")
    dataset = report.get("dataset", {})
    lines.append(f"- Fuente: `{dataset.get('source_path', 'synthetic/in-memory')}`")
    lines.append(f"- Shape: `{dataset.get('shape', 'n/a')}`")
    lines.append(f"- Columnas usadas: `{dataset.get('columns', [])}`")
    lines.append("")
    lines.append("## 3. Ranking topológico")
    lines.append("| rank | manifold | score | recon_error | std | smoothness | utilization |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for e in report.get("evaluations", []):
        lines.append(
            f"| {e.get('rank')} | {e.get('manifold')} | {e.get('score_mean', e.get('score')):.6g} | "
            f"{e.get('reconstruction_error_mean', e.get('reconstruction_error')):.6g} | "
            f"{e.get('reconstruction_error_std', 0.0):.6g} | {e.get('smoothness'):.6g} | {e.get('latent_utilization'):.6g} |"
        )
    lines.append("")
    lines.append("## 4. Visualizaciones")
    for name, path in figures.items():
        rel = Path(path).relative_to(output_dir)
        lines.append(f"### {name}")
        lines.append(f"![{name}]({rel.as_posix()})")
        lines.append("")
    lines.append("## 5. Interpretación")
    summary = agent_summary(report)
    lines.append(summary["one_sentence"])
    lines.append("")
    lines.append(f"Acción recomendada: {summary['recommended_next_action']}")
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_html(markdown_path: str, output_dir: Path) -> str:
    md = Path(markdown_path).read_text(encoding="utf-8")
    html = "<html><head><meta charset='utf-8'><title>ESO Report</title>"
    html += "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.5}img{max-width:100%;border:1px solid #ddd}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}</style>"
    html += "</head><body><pre style='white-space:pre-wrap'>"
    html += md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += "</pre></body></html>"
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def write_report_bundle(report: dict, data, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    figures = generate_figures(data, report, figures_dir)
    report = dict(report)
    report["schema_version"] = "eso.report.v1"
    report["figures"] = figures
    report["agent_summary"] = agent_summary(report)

    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    diagnosis_path = output_dir / "diagnosis.json"
    diagnosis_path.write_text(json.dumps(_json_safe(report.get("diagnosis", {})), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = write_metrics(report, output_dir)
    md_path = write_markdown(report, figures, output_dir)
    html_path = write_html(md_path, output_dir)

    return {
        "report_json": str(json_path),
        "diagnosis_json": str(diagnosis_path),
        "metrics_csv": metrics_path,
        "report_md": md_path,
        "report_html": html_path,
        "figures": figures,
    }
