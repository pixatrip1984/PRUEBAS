import numpy as np
import pandas as pd

from eso.data.loader import load_dataset


def test_load_dataset_csv_with_columns(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"t": [0, 1, 2, 3], "x": [1.0, 2.0, np.nan, 4.0], "label": ["a", "b", "c", "d"]}).to_csv(path, index=False)
    loaded = load_dataset(str(path), columns=["x"], normalize_method="none")
    assert loaded.data.shape == (4, 1)
    assert np.isfinite(loaded.data).all()
    assert loaded.columns == ["x"]
    assert loaded.validation.rows == 4


def test_loaded_dataset_info_has_stable_fingerprint(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}).to_csv(path, index=False)
    a = load_dataset(str(path), columns=["x", "y"], normalize_method="none").info()
    b = load_dataset(str(path), columns=["x", "y"], normalize_method="none").info()
    assert len(a["data_sha256"]) == 64
    assert a["data_sha256"] == b["data_sha256"]
