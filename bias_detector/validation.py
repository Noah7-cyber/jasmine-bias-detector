"""Input validation and data-source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    missing_by_column: dict[str, int]
    missing_selected: dict[str, int]
    warnings: tuple[str, ...]


def inspect_frame(frame: pd.DataFrame, max_rows: int = 500_000) -> ValidationReport:
    warnings: list[str] = []
    if frame.empty:
        warnings.append("The file contains no data rows.")
    if len(frame) > max_rows:
        warnings.append(f"This file has {len(frame):,} rows; the current upload limit is {max_rows:,}.")
    if frame.columns.duplicated().any():
        dupes = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        warnings.append("Duplicate column names: " + ", ".join(dupes))
    return ValidationReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_by_column={str(c): int(v) for c, v in frame.isna().sum().items()},
        missing_selected={},
        warnings=tuple(warnings),
    )


def selected_missing_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {c: int(frame[c].isna().sum()) for c in columns if c in frame.columns}


def binary_values(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())


def validate_mapping(values: list[str], positive: str, negative: str, label: str) -> None:
    if not positive or not negative or positive == negative:
        raise ValueError(f"Choose different positive and negative values for {label}.")
    invalid = sorted(set(values) - {positive, negative})
    if invalid:
        raise ValueError(f"{label} contains values outside the chosen binary mapping: {', '.join(invalid[:8])}.")
