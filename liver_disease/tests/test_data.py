"""Tests for ILPD dataset loading, provenance, schema validation, and audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liver_disease.config import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from liver_disease.data.loader import (
    ensure_raw_dataset,
    get_dataset_provenance,
    load_raw_dataset,
    normalize_inference_columns,
    split_features_and_target,
)
from liver_disease.data.validation import (
    run_comprehensive_data_audit,
    validate_feature_dataframe,
)


def test_raw_dataset_loads_with_expected_schema_and_provenance() -> None:
    raw_path = ensure_raw_dataset()
    assert raw_path.exists()

    prov = get_dataset_provenance(raw_path)
    assert prov["size_bytes"] == 23817
    assert len(str(prov["sha256"])) == 64

    df = load_raw_dataset(raw_path)
    assert df.shape == (583, 11)
    assert list(df.columns) == RAW_FEATURE_COLUMNS + [TARGET_COLUMN]


def test_split_features_and_target_mapping() -> None:
    df = load_raw_dataset()
    X, y = split_features_and_target(df)
    assert list(X.columns) == RAW_FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == int((df[TARGET_COLUMN] == 1).sum())
    assert int((y == 0).sum()) == int((df[TARGET_COLUMN] == 2).sum())


def test_split_features_rejects_invalid_target() -> None:
    df = load_raw_dataset().head(5).copy()
    df.loc[0, TARGET_COLUMN] = 99
    with pytest.raises(ValueError, match="Invalid target values"):
        split_features_and_target(df)


def test_data_audit_verified_counts() -> None:
    df = load_raw_dataset()
    audit = run_comprehensive_data_audit(df)
    assert audit["shape"] == {"rows": 583, "columns": 11}
    assert audit["missing_values"]["total_missing_cells"] == 4
    assert "A/G Ratio" in audit["missing_values"]["by_column"]
    assert audit["duplicates"]["exact_duplicate_extra_rows"] == 13
    assert len(audit["duplicates"]["feature_duplicate_target_conflicts"]) == 0
    assert audit["target_distribution"]["liver_disease_count"] == 416
    assert audit["target_distribution"]["no_liver_disease_count"] == 167


def test_validate_feature_dataframe_catches_invalid_inputs() -> None:
    df_valid = load_raw_dataset()[RAW_FEATURE_COLUMNS].dropna().head(5).copy()
    assert validate_feature_dataframe(df_valid, allow_missing=False) == []

    # Invalid gender
    df_bad_gender = df_valid.copy()
    df_bad_gender.loc[df_bad_gender.index[0], "Gender"] = "Unknown"
    errs = validate_feature_dataframe(df_bad_gender)
    assert any("Gender" in e for e in errs)

    # Out-of-range TB
    df_bad_range = df_valid.copy()
    df_bad_range.loc[df_bad_range.index[0], "TB"] = -2.5
    errs = validate_feature_dataframe(df_bad_range)
    assert any("TB" in e for e in errs)

    # Missing value disallowed
    df_missing = df_valid.copy()
    df_missing.loc[df_missing.index[0], "A/G Ratio"] = np.nan
    errs = validate_feature_dataframe(df_missing, allow_missing=False)
    assert any("missing" in e.lower() for e in errs)


def test_normalize_inference_columns_rejects_extra_columns() -> None:
    df_bad = pd.DataFrame(
        [
            {
                "age": 45,
                "gender": "Male",
                "total_bilirubin": 1.0,
                "direct_bilirubin": 0.3,
                "alkaline_phosphotase": 200,
                "alamine_aminotransferase": 35,
                "aspartate_aminotransferase": 40,
                "total_proteins": 6.8,
                "albumin": 3.4,
                "albumin_and_globulin_ratio": 1.0,
                "unexpected_col": 123,
            }
        ]
    )
    with pytest.raises(ValueError, match="Unexpected columns"):
        normalize_inference_columns(df_bad)
