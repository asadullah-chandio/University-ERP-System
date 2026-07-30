"""Excel/CSV upload reading and validation against the canonical student schema.

Design principle: fail loudly on things that would silently corrupt
predictions (missing StudentID, no usable columns at all), but degrade
gracefully on everything else (an optional column missing just means one
fewer module runs; an out-of-range value gets clipped and counted, not
rejected) -- so "any university" can use this with whatever subset of data
they actually have.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.schema import (
    ALL_TEMPLATE_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODULE_REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file's extension isn't csv/xlsx/xls."""


@dataclass
class ValidationReport:
    """Result of validating an uploaded student dataframe.

    Attributes:
        errors: Blocking problems -- the file cannot be processed at all.
        warnings: Non-blocking problems -- processing continues, with the
            affected rows/columns/modules called out.
        available_modules: Module keys (from schema.MODULE_REQUIRED_COLUMNS)
            whose required columns are all present.
        missing_columns: Template columns not present in the upload at all.
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    available_modules: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a dataframe.

    Args:
        file_bytes: Raw file content.
        filename: Original filename, used only to detect the format.

    Returns:
        The parsed dataframe (columns not yet validated).

    Raises:
        UnsupportedFileTypeError: If the extension isn't csv/xlsx/xls.
        ValueError: If the file can't be parsed (corrupt/empty/wrong format).
    """
    lower_name = filename.lower()
    if not lower_name.endswith(SUPPORTED_EXTENSIONS):
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{filename}'. Please upload a .csv, .xlsx, or .xls file."
        )

    buffer = io.BytesIO(file_bytes)
    try:
        if lower_name.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': the file may be corrupt or in an unexpected format ({exc}).") from exc

    if df.empty:
        raise ValueError(f"'{filename}' contains no rows.")

    df.columns = df.columns.str.strip()
    return df


def validate_upload(df: pd.DataFrame) -> ValidationReport:
    """Validate an uploaded student dataframe against the canonical schema.

    Args:
        df: Raw uploaded dataframe (post `read_uploaded_file`).

    Returns:
        A ValidationReport. Even a "valid" report may contain warnings.
    """
    report = ValidationReport()
    present_columns = set(df.columns)

    for col in IDENTIFIER_COLUMNS:
        if col not in present_columns:
            report.errors.append(f"Required column '{col}' is missing.")
        elif df[col].isna().any():
            report.errors.append(f"Column '{col}' has missing values -- every row must have a unique student ID.")
        elif df[col].duplicated().any():
            n_dupes = int(df[col].duplicated().sum())
            report.errors.append(f"Column '{col}' has {n_dupes} duplicate value(s) -- student IDs must be unique.")

    if not report.is_valid:
        return report

    report.missing_columns = [c for c in ALL_TEMPLATE_COLUMNS if c not in present_columns]

    for col, (low, high) in NUMERIC_COLUMNS.items():
        if col not in present_columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(coerced.isna().sum() - df[col].isna().sum())
        if n_bad > 0:
            report.warnings.append(f"Column '{col}' has {n_bad} non-numeric value(s); those rows will be treated as missing for this column.")
        n_missing = int(coerced.isna().sum())
        if n_missing > 0:
            report.warnings.append(f"Column '{col}' has {n_missing} missing value(s); they will be filled with the column median.")
        out_of_range = coerced.dropna()[(coerced.dropna() < low) | (coerced.dropna() > high)]
        if len(out_of_range) > 0:
            report.warnings.append(
                f"Column '{col}' has {len(out_of_range)} value(s) outside the expected range ({low}-{high}); they will be clipped."
            )

    for col, allowed in CATEGORICAL_COLUMNS.items():
        if col not in present_columns:
            continue
        unknown = set(df[col].dropna().astype(str).unique()) - set(allowed)
        if unknown:
            report.warnings.append(
                f"Column '{col}' has unrecognized value(s) {sorted(unknown)} (expected one of {allowed}); "
                "they will be mapped to the most common category."
            )

    for col in DATE_COLUMNS:
        if col not in present_columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_bad = int(parsed.isna().sum() - df[col].isna().sum())
        if n_bad > 0:
            report.warnings.append(f"Column '{col}' has {n_bad} value(s) that aren't valid dates; they will be excluded from forecasting.")

    for module, required_cols in MODULE_REQUIRED_COLUMNS.items():
        if all(c in present_columns for c in required_cols):
            report.available_modules.append(module)
        else:
            missing = [c for c in required_cols if c not in present_columns]
            report.warnings.append(f"Module '{module}' will be skipped -- missing column(s): {missing}.")

    if not report.available_modules:
        report.errors.append(
            "None of the 6 prediction modules have enough columns to run. "
            "Please check the sample template for the required columns."
        )

    return report


def clean_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce, clip, and impute an already-validated dataframe for prediction.

    Assumes `validate_upload` has already been run and returned no errors.

    Args:
        df: Raw uploaded dataframe.

    Returns:
        A cleaned copy: numeric columns coerced/clipped/imputed, unknown
        categorical values left as-is (handled downstream by the fallback
        encoders in `src.encoding`), dates parsed.
    """
    df = df.copy()
    df["StudentID"] = df["StudentID"].astype(str).str.strip()

    for col, (low, high) in NUMERIC_COLUMNS.items():
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            median = coerced.median()
            coerced = coerced.fillna(median if pd.notna(median) else (low + high) / 2)
        df[col] = coerced.clip(lower=low, upper=high)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
