"""Dataset validation for ESO agent-ready ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataValidationReport:
    rows: int
    columns: int
    numeric_columns: list[str]
    non_numeric_columns: list[str]
    selected_columns: list[str]
    missing_ratio_by_column: dict[str, float]
    constant_columns: list[str] = field(default_factory=list)
    highly_correlated_columns: list[tuple[str, str, float]] = field(default_factory=list)
    recommended_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["highly_correlated_columns"] = [list(x) for x in self.highly_correlated_columns]
        return out


def validate_dataframe(df: pd.DataFrame, columns: list[str] | None = None, corr_threshold: float = 0.98) -> DataValidationReport:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric]
    warnings: list[str] = []

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        selected = [c for c in columns if c in numeric]
        skipped = [c for c in columns if c not in numeric]
        if skipped:
            warnings.append(f"non numeric selected columns skipped: {skipped}")
    else:
        selected = numeric

    if not selected:
        raise ValueError("no numeric columns available for ESO")

    missing_ratio = {c: float(df[c].isna().mean()) for c in df.columns}
    constant = []
    for c in selected:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            constant.append(c)
    if constant:
        warnings.append(f"constant columns detected: {constant}")

    high_corr: list[tuple[str, str, float]] = []
    if len(selected) > 1:
        corr = df[selected].corr(numeric_only=True).abs()
        for i, a in enumerate(selected):
            for b in selected[i + 1 :]:
                val = corr.loc[a, b]
                if np.isfinite(val) and float(val) >= corr_threshold:
                    high_corr.append((a, b, float(val)))
        if high_corr:
            warnings.append("highly correlated columns detected")

    recommended = [c for c in selected if c not in constant]
    return DataValidationReport(
        rows=int(len(df)),
        columns=int(len(df.columns)),
        numeric_columns=numeric,
        non_numeric_columns=non_numeric,
        selected_columns=selected,
        missing_ratio_by_column=missing_ratio,
        constant_columns=constant,
        highly_correlated_columns=high_corr,
        recommended_columns=recommended or selected,
        warnings=warnings,
    )
