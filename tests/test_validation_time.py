import pandas as pd

from eso.data.validation import validate_dataframe


def test_validate_dataframe_profiles_timestamp_column():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h").astype(str),
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    report = validate_dataframe(df)
    payload = report.to_dict()
    assert payload["time_columns"] == ["timestamp"]
    assert payload["time_summary"]["timestamp"]["monotonic_increasing"] is True
    assert payload["time_summary"]["timestamp"]["median_delta_seconds"] == 3600.0


def test_validate_dataframe_warns_on_non_monotonic_time():
    df = pd.DataFrame({
        "timestamp": ["2024-01-02", "2024-01-01", "2024-01-01"],
        "x": [1.0, 2.0, 3.0],
    })
    report = validate_dataframe(df)
    warnings = " ".join(report.warnings)
    assert "not monotonic" in warnings
    assert "duplicate" in warnings
