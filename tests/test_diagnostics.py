import numpy as np

from eso.diagnostics.report import run_diagnosis


def test_harmonic_diagnosis_detects_periodicity():
    t = np.linspace(0, 8 * np.pi, 256)
    data = np.stack([np.cos(t), np.sin(t)], axis=1)
    report = run_diagnosis(data)
    assert report["dimension"]["consensus_dimension"] > 0
    assert report["symmetries"]["periodic_hint"] is True
    assert "summary" in report
