"""Tests for leakage-free splitting, feature engineering, and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from liver_disease.data.loader import load_raw_dataset
from liver_disease.data.preprocessing import (
    build_preprocessor,
    build_training_pipeline,
    create_leakage_free_split,
    deduplicate_dataset,
    get_transformed_feature_names,
)


def test_deduplication_and_leakage_free_split() -> None:
    df_raw = load_raw_dataset()
    df_dedup, dedup_info = deduplicate_dataset(df_raw)
    assert len(df_dedup) == 570
    assert dedup_info["exact_duplicates_removed"] == 13
    assert dedup_info["feature_duplicate_target_conflicts"] == 0

    X_train, X_test, y_train, y_test, meta = create_leakage_free_split(
        df_raw, test_size=0.20, random_state=42, save_artifacts=False
    )
    assert len(X_train) == 456
    assert len(X_test) == 114
    assert meta["train_test_feature_overlap"] == 0
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_preprocessor_handles_missing_values_and_no_leakage() -> None:
    df_raw = load_raw_dataset()
    X_train, X_test, y_train, y_test, _ = create_leakage_free_split(
        df_raw, save_artifacts=False
    )

    pre = build_preprocessor(scaler="robust", add_clinical_ratios=False, log_transform_skewed=True)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    assert not np.isnan(X_train_t).any()
    assert not np.isnan(X_test_t).any()

    feat_names = get_transformed_feature_names(pre)
    assert len(feat_names) == X_train_t.shape[1] == 10
    assert any("Gender" in f for f in feat_names)


def test_preprocessor_with_clinical_ratios() -> None:
    df_raw = load_raw_dataset()
    X_train, X_test, _, _, _ = create_leakage_free_split(df_raw, save_artifacts=False)

    pre = build_preprocessor(scaler="robust", add_clinical_ratios=True, log_transform_skewed=True)
    X_train_t = pre.fit_transform(X_train)
    feat_names = get_transformed_feature_names(pre)
    assert X_train_t.shape[1] == 14
    assert "AST_ALT_Ratio" in feat_names
    assert "DB_TB_Ratio" in feat_names


def test_in_fold_oversampling_pipelines_fit_and_predict() -> None:
    df_raw = load_raw_dataset()
    X_train, X_test, y_train, _, _ = create_leakage_free_split(df_raw, save_artifacts=False)

    for strategy in ("none", "random_oversampler", "smote"):
        pipe = build_training_pipeline(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            scaler="robust",
            add_clinical_ratios=False,
            log_transform_skewed=True,
            imbalance_strategy=strategy,  # type: ignore[arg-type]
        )
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
