"""Dataset acquisition, provenance verification, and loading for ILPD."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

from liver_disease.config import (
    API_TO_RAW_COLUMN_MAP,
    RAW_DATASET_PATH,
    RAW_FEATURE_COLUMNS,
    ROOT_DATASET_FALLBACK,
    TARGET_COLUMN,
    TARGET_DISEASE_RAW,
    TARGET_NO_DISEASE_RAW,
    ensure_project_directories,
)
from liver_disease.utils.io import compute_sha256

logger = logging.getLogger(__name__)

EXPECTED_RAW_COLUMNS = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]


def ensure_raw_dataset(source_path: Path | None = None) -> Path:
    """Ensure the canonical raw ILPD CSV exists inside data/raw/ without modifying original.

    Returns the path to the canonical raw dataset file.
    """
    ensure_project_directories()
    if RAW_DATASET_PATH.exists():
        return RAW_DATASET_PATH

    candidate = source_path or ROOT_DATASET_FALLBACK
    if not candidate.exists():
        raise FileNotFoundError(
            f"Raw ILPD dataset not found at {RAW_DATASET_PATH} or {candidate}. "
            "Please place 'Indian Liver Patient Dataset (ILPD).csv' in data/raw/."
        )

    shutil.copy2(candidate, RAW_DATASET_PATH)
    logger.info("Copied raw ILPD dataset from %s to %s", candidate, RAW_DATASET_PATH)
    return RAW_DATASET_PATH


def get_dataset_provenance(path: Path | None = None) -> dict[str, object]:
    """Return file metadata and SHA-256 checksum for reproducibility tracking."""
    dataset_path = path or ensure_raw_dataset()
    stat = dataset_path.stat()
    return {
        "filename": dataset_path.name,
        "path": str(dataset_path),
        "size_bytes": stat.st_size,
        "sha256": compute_sha256(dataset_path),
        "source_name": "UCI Machine Learning Repository — ILPD (Indian Liver Patient Dataset)",
        "source_url": "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        "original_creators": "Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu (2012)",
        "geographic_origin": "North East of Andhra Pradesh, India",
    }


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw ILPD CSV and verify exact column schema."""
    dataset_path = path or ensure_raw_dataset()
    df = pd.read_csv(dataset_path)

    if list(df.columns) != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            f"Unexpected ILPD columns.\nExpected: {EXPECTED_RAW_COLUMNS}\nFound:    {list(df.columns)}"
        )

    # Strip accidental whitespace in string columns without altering medical values
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()

    logger.info("Loaded raw ILPD dataset: shape=%s from %s", df.shape, dataset_path)
    return df


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split ILPD dataframe into predictor matrix X and binary target y.

    Target mapping (explicit and verified):
        Selector == 1 (Liver Disease)    -> y = 1 (positive class)
        Selector == 2 (No Liver Disease) -> y = 0 (negative class)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    unique_targets = set(df[TARGET_COLUMN].dropna().unique().tolist())
    expected_targets = {TARGET_DISEASE_RAW, TARGET_NO_DISEASE_RAW}
    if not unique_targets.issubset(expected_targets):
        raise ValueError(
            f"Invalid target values in '{TARGET_COLUMN}': {unique_targets}. "
            f"Expected subset of {expected_targets}."
        )
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    X = df[RAW_FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN].astype(int) == TARGET_DISEASE_RAW).astype(int)
    y.name = "liver_disease"
    return X, y


def normalize_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either canonical API snake_case columns or raw ILPD CSV columns.

    Strictly rejects missing required columns or unexpected extra columns.
    """
    cols = [str(c).strip() for c in df.columns]
    df_clean = df.copy()
    df_clean.columns = cols

    raw_set = set(RAW_FEATURE_COLUMNS)
    api_set = set(API_TO_RAW_COLUMN_MAP.keys())
    col_set = set(cols)

    if col_set == raw_set:
        return df_clean[RAW_FEATURE_COLUMNS].copy()

    if col_set == api_set:
        renamed = df_clean.rename(columns=API_TO_RAW_COLUMN_MAP)
        return renamed[RAW_FEATURE_COLUMNS].copy()

    # Check if there are unexpected or missing columns
    # Decide which schema is closer for a clearer error message
    if len(col_set & api_set) > len(col_set & raw_set):
        missing = sorted(api_set - col_set)
        extra = sorted(col_set - api_set)
    else:
        missing = sorted(raw_set - col_set)
        extra = sorted(col_set - raw_set)

    raise ValueError(
        f"Invalid feature schema. Missing columns: {missing}; Unexpected columns: {extra}. "
        f"Accepted schemas: raw ILPD {RAW_FEATURE_COLUMNS} or canonical API {list(API_TO_RAW_COLUMN_MAP.keys())}."
    )
