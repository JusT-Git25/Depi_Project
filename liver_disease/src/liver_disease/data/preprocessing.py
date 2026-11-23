"""Leakage-free data splitting and reproducible preprocessing pipelines for ILPD."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from liver_disease.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_project_directories,
)
from liver_disease.data.loader import split_features_and_target
from liver_disease.features.engineering import ClinicalFeatureEngineer
from liver_disease.utils.io import save_json

logger = logging.getLogger(__name__)

ScalerType = Literal["robust", "standard", "none"]
ImbalanceStrategy = Literal["none", "class_weight", "random_oversampler", "smote"]


def deduplicate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove exact duplicate observations while documenting every dropped index.

    Exact duplicates in ILPD would leak identical patients across train/test splits
    and artificially inflate validation metrics if retained.
    """
    n_before = len(df)
    dup_mask = df.duplicated(keep="first")
    dropped_indices = df.index[dup_mask].tolist()

    # Verify zero target conflicts on feature-only duplicates before proceeding
    feat_dup_mask = df.duplicated(subset=RAW_FEATURE_COLUMNS, keep=False)
    conflicts = 0
    if feat_dup_mask.any():
        for _, grp in df[feat_dup_mask].groupby(RAW_FEATURE_COLUMNS, dropna=False):
            if grp[TARGET_COLUMN].nunique() > 1:
                conflicts += 1

    dedup_df = df.loc[~dup_mask].copy().reset_index(drop=True)
    summary = {
        "rows_before": n_before,
        "exact_duplicates_removed": int(dup_mask.sum()),
        "dropped_raw_row_indices": dropped_indices,
        "feature_duplicate_target_conflicts": conflicts,
        "rows_after": len(dedup_df),
    }
    logger.info(
        "Deduplication: %d -> %d rows (%d exact duplicates removed, %d target conflicts).",
        n_before,
        len(dedup_df),
        int(dup_mask.sum()),
        conflicts,
    )
    return dedup_df, summary


def create_leakage_free_split(
    df_raw: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Create a deterministic, leakage-free stratified train/test split.

    Steps:
    1. Deduplicate exact duplicate rows so identical observations cannot straddle train/test.
    2. Perform stratified train/test split preserving class proportions.
    3. Verify zero exact row overlap between X_train and X_test.
    4. Optionally persist split CSVs and metadata to data/processed/.
    """
    df_clean, dedup_info = deduplicate_dataset(df_raw)
    X, y = split_features_and_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Leakage verification: zero identical feature rows across train & test
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False).tolist())
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).tolist())
    overlap_count = len(train_hashes & test_hashes)
    if overlap_count > 0:
        raise RuntimeError(f"Data leakage detected: {overlap_count} overlapping feature rows between train and test!")

    split_meta: Dict[str, Any] = {
        "random_state": random_state,
        "test_size": test_size,
        "deduplication": dedup_info,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_count": int(y_train.sum()),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_count": int(y_test.sum()),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "train_test_feature_overlap": overlap_count,
        "train_indices_post_dedup": X_train.index.tolist(),
        "test_indices_post_dedup": X_test.index.tolist(),
    }

    if save_artifacts:
        ensure_project_directories()
        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train.map({1: 1, 0: 2})
        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test.map({1: 1, 0: 2})

        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        save_json(split_meta, PROCESSED_DATA_DIR / "split_metadata.json")
        logger.info("Saved leakage-free train/test split to %s", PROCESSED_DATA_DIR)

    return X_train, X_test, y_train, y_test, split_meta


def get_expected_numeric_columns(add_clinical_ratios: bool = True) -> List[str]:
    """Return numeric feature column names after optional clinical feature engineering."""
    cols = list(NUMERIC_FEATURE_COLUMNS)
    if add_clinical_ratios:
        cols.extend(["AST_ALT_Ratio", "DB_TB_Ratio", "Indirect_Bilirubin", "Globulin"])
    return cols


def build_preprocessor(
    scaler: ScalerType = "robust",
    add_clinical_ratios: bool = True,
    log_transform_skewed: bool = True,
) -> Pipeline:
    """Build a leakage-free scikit-learn preprocessing pipeline.

    All stateful steps (median imputation, scaling, one-hot encoding) are fitted
    exclusively on training data inside cross-validation or final training.
    """
    numeric_cols = get_expected_numeric_columns(add_clinical_ratios=add_clinical_ratios)
    categorical_cols = list(CATEGORICAL_FEATURE_COLUMNS)

    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler == "none":
        pass
    else:
        raise ValueError(f"Unsupported scaler '{scaler}'. Choose from 'robust', 'standard', 'none'.")

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[["Female", "Male"]],
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineer",
                ClinicalFeatureEngineer(
                    add_clinical_ratios=add_clinical_ratios,
                    log_transform_skewed=log_transform_skewed,
                ),
            ),
            ("column_transformer", col_transformer),
        ]
    )


def build_training_pipeline(
    estimator: Any,
    scaler: ScalerType = "robust",
    add_clinical_ratios: bool = True,
    log_transform_skewed: bool = True,
    imbalance_strategy: ImbalanceStrategy = "none",
    random_state: int = RANDOM_STATE,
) -> Pipeline | ImbPipeline:
    """Assemble full preprocessing + optional in-fold resampling + estimator pipeline.

    Note:
    - When `imbalance_strategy` is `"random_oversampler"` or `"smote"`, an
      `imblearn.pipeline.Pipeline` is used so resampling runs ONLY during `.fit()`
      on training folds and is automatically bypassed during `.predict()` / `.predict_proba()`.
    """
    preprocessor = build_preprocessor(
        scaler=scaler,
        add_clinical_ratios=add_clinical_ratios,
        log_transform_skewed=log_transform_skewed,
    )

    if imbalance_strategy in {"random_oversampler", "smote"}:
        # imblearn.pipeline.Pipeline disallows nested Pipeline objects as intermediate steps;
        # flatten preprocessor steps cleanly while preserving identical transformation semantics.
        fe_step = preprocessor.named_steps["feature_engineer"]
        ct_step = preprocessor.named_steps["column_transformer"]
        sampler = (
            RandomOverSampler(random_state=random_state)
            if imbalance_strategy == "random_oversampler"
            else SMOTE(random_state=random_state, k_neighbors=5)
        )
        return ImbPipeline(
            steps=[
                ("feature_engineer", fe_step),
                ("column_transformer", ct_step),
                ("sampler", sampler),
                ("classifier", estimator),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def get_transformed_feature_names(fitted_pipeline_or_preprocessor: Any) -> List[str]:
    """Extract clean output feature names from a fitted preprocessor or full pipeline."""
    if hasattr(fitted_pipeline_or_preprocessor, "named_steps"):
        steps = fitted_pipeline_or_preprocessor.named_steps
        if "column_transformer" in steps:
            col_transformer = steps["column_transformer"]
        elif "preprocessor" in steps:
            col_transformer = steps["preprocessor"].named_steps["column_transformer"]
        else:
            raise ValueError("Could not locate 'column_transformer' in pipeline.")
    else:
        col_transformer = fitted_pipeline_or_preprocessor
    names = col_transformer.get_feature_names_out()
    return [str(n) for n in names]
