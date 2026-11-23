"""Comprehensive data audit and input validation for ILPD."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from liver_disease.config import (
    FEATURE_SPECS,
    NUMERIC_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_DISEASE_RAW,
    TARGET_NO_DISEASE_RAW,
    VALID_GENDERS,
)


def validate_feature_dataframe(df: pd.DataFrame, allow_missing: bool = False) -> List[str]:
    """Validate predictor dataframe types, categories, ranges, and NaNs.

    Returns a list of validation error messages (empty list means valid).
    """
    errors: List[str] = []

    missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in RAW_FEATURE_COLUMNS]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")
    if errors:
        return errors

    if not allow_missing:
        null_counts = df[RAW_FEATURE_COLUMNS].isna().sum()
        for col, cnt in null_counts.items():
            if cnt > 0:
                errors.append(f"Column '{col}' contains {int(cnt)} missing value(s).")

    # Categorical check
    invalid_genders = set(df["Gender"].dropna().astype(str).str.strip().unique()) - set(VALID_GENDERS)
    if invalid_genders:
        errors.append(f"Column 'Gender' contains invalid values: {sorted(invalid_genders)}. Expected {VALID_GENDERS}.")

    # Numeric checks
    for col in NUMERIC_FEATURE_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        non_numeric_mask = df[col].notna() & series.isna()
        if non_numeric_mask.any():
            errors.append(f"Column '{col}' has {int(non_numeric_mask.sum())} non-numeric value(s).")
            continue

        valid = series.dropna()
        if valid.empty:
            continue

        if np.isinf(valid).any():
            errors.append(f"Column '{col}' contains infinite values.")

        spec = FEATURE_SPECS[col]
        if spec.min_value is not None and (valid < spec.min_value).any():
            errors.append(
                f"Column '{col}' has {int((valid < spec.min_value).sum())} value(s) below allowed minimum {spec.min_value} {spec.unit}."
            )
        if spec.max_value is not None and (valid > spec.max_value).any():
            errors.append(
                f"Column '{col}' has {int((valid > spec.max_value).sum())} value(s) above allowed maximum {spec.max_value} {spec.unit}."
            )

    return errors


def run_comprehensive_data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute a thorough, transparent audit of the raw ILPD dataset."""
    n_rows, n_cols = df.shape

    # 1. Missing values
    missing_summary: Dict[str, Any] = {}
    for col in df.columns:
        na_mask = df[col].isna()
        count = int(na_mask.sum())
        if count > 0:
            missing_summary[col] = {
                "count": count,
                "percentage": round(100.0 * count / n_rows, 4),
                "row_indices": df.index[na_mask].tolist(),
            }

    # 2. Target distribution
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    disease_count = int(target_counts.get(TARGET_DISEASE_RAW, 0))
    no_disease_count = int(target_counts.get(TARGET_NO_DISEASE_RAW, 0))
    imbalance_ratio = round(disease_count / max(no_disease_count, 1), 4)

    # 3. Exact duplicates (all columns) and feature-only duplicates
    exact_dup_mask = df.duplicated(keep=False)
    exact_dup_extra = int(df.duplicated(keep="first").sum())
    exact_dup_groups: List[Dict[str, Any]] = []
    if exact_dup_mask.any():
        for _, grp in df[exact_dup_mask].groupby(list(df.columns), dropna=False):
            exact_dup_groups.append(
                {
                    "row_indices": grp.index.tolist(),
                    "count": int(len(grp)),
                    "selector": int(grp[TARGET_COLUMN].iloc[0]),
                    "age": int(grp["Age"].iloc[0]),
                    "gender": str(grp["Gender"].iloc[0]),
                }
            )

    feat_dup_mask = df.duplicated(subset=RAW_FEATURE_COLUMNS, keep=False)
    feat_dup_extra = int(df.duplicated(subset=RAW_FEATURE_COLUMNS, keep="first").sum())
    target_conflicts: List[Dict[str, Any]] = []
    if feat_dup_mask.any():
        for _, grp in df[feat_dup_mask].groupby(RAW_FEATURE_COLUMNS, dropna=False):
            if grp[TARGET_COLUMN].nunique() > 1:
                target_conflicts.append(
                    {
                        "row_indices": grp.index.tolist(),
                        "targets": grp[TARGET_COLUMN].tolist(),
                    }
                )

    # 4. Near-duplicates (non-identical rows sharing Age + Gender and very small standardized distance)
    dedup_df = df.drop_duplicates(keep="first").reset_index()
    num_vals = dedup_df[NUMERIC_FEATURE_COLUMNS].copy()
    num_vals = num_vals.fillna(num_vals.median())
    stds = num_vals.std().replace(0, 1.0)
    z_vals = ((num_vals - num_vals.mean()) / stds).to_numpy()
    dist_mat = cdist(z_vals, z_vals, metric="euclidean")
    np.fill_diagonal(dist_mat, np.inf)

    near_dup_pairs: List[Dict[str, Any]] = []
    close_i, close_j = np.where(dist_mat < 0.25)
    for i, j in zip(close_i, close_j):
        if i < j and dedup_df.loc[i, "Gender"] == dedup_df.loc[j, "Gender"]:
            near_dup_pairs.append(
                {
                    "row_index_1": int(dedup_df.loc[i, "index"]),
                    "row_index_2": int(dedup_df.loc[j, "index"]),
                    "standardized_euclidean_distance": round(float(dist_mat[i, j]), 4),
                    "same_target": bool(dedup_df.loc[i, TARGET_COLUMN] == dedup_df.loc[j, TARGET_COLUMN]),
                }
            )

    # 5. Physiological / domain consistency checks
    db_gt_tb = df[df["DB"] > df["TB"]]
    alb_gt_tp = df[df["ALB"] > df["TP"]]
    non_positive_checks = {}
    for col in NUMERIC_FEATURE_COLUMNS:
        non_pos = int((df[col].dropna() <= 0).sum())
        if non_pos > 0:
            non_positive_checks[col] = non_pos

    # Check A/G Ratio consistency against ALB / (TP - ALB)
    globulin = df["TP"] - df["ALB"]
    valid_ag = df["A/G Ratio"].notna() & (globulin > 0)
    calc_ag = df.loc[valid_ag, "ALB"] / globulin.loc[valid_ag]
    ag_abs_diff = (df.loc[valid_ag, "A/G Ratio"] - calc_ag).abs()
    ag_large_diff_indices = df.loc[valid_ag].index[ag_abs_diff > 0.25].tolist()

    # 6. Numerical summary & IQR / skewness / outlier analysis
    numeric_summary: Dict[str, Any] = {}
    for col in NUMERIC_FEATURE_COLUMNS:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        n_outliers_iqr = int(((s < lower_fence) | (s > upper_fence)).sum())
        numeric_summary[col] = {
            "count": int(s.count()),
            "missing": int(df[col].isna().sum()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "q25": round(q1, 4),
            "median": round(float(s.median()), 4),
            "q75": round(q3, 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "iqr_lower_fence": round(lower_fence, 4),
            "iqr_upper_fence": round(upper_fence, 4),
            "iqr_outlier_count": n_outliers_iqr,
            "iqr_outlier_pct": round(100.0 * n_outliers_iqr / len(s), 2),
        }

    # 7. Categorical summary
    gender_counts = df["Gender"].value_counts(dropna=False).to_dict()
    gender_by_target = (
        pd.crosstab(df["Gender"], df[TARGET_COLUMN])
        .rename(columns={1: "liver_disease_1", 2: "no_liver_disease_2"})
        .to_dict(orient="index")
    )

    return {
        "shape": {"rows": n_rows, "columns": n_cols},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            "total_missing_cells": int(df.isna().sum().sum()),
            "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
            "by_column": missing_summary,
        },
        "target_distribution": {
            "raw_counts": {str(k): int(v) for k, v in target_counts.items()},
            "liver_disease_count": disease_count,
            "liver_disease_pct": round(100.0 * disease_count / n_rows, 2),
            "no_liver_disease_count": no_disease_count,
            "no_liver_disease_pct": round(100.0 * no_disease_count / n_rows, 2),
            "imbalance_ratio_pos_to_neg": imbalance_ratio,
        },
        "duplicates": {
            "exact_duplicate_extra_rows": exact_dup_extra,
            "exact_duplicate_total_rows_involved": int(exact_dup_mask.sum()),
            "exact_duplicate_groups_count": len(exact_dup_groups),
            "exact_duplicate_groups": exact_dup_groups,
            "feature_only_duplicate_extra_rows": feat_dup_extra,
            "feature_duplicate_target_conflicts": target_conflicts,
            "near_duplicate_pairs_z_lt_0_25": near_dup_pairs,
        },
        "domain_consistency_checks": {
            "direct_gt_total_bilirubin_count": int(len(db_gt_tb)),
            "direct_gt_total_bilirubin_indices": db_gt_tb.index.tolist(),
            "albumin_gt_total_proteins_count": int(len(alb_gt_tp)),
            "albumin_gt_total_proteins_indices": alb_gt_tp.index.tolist(),
            "non_positive_values": non_positive_checks,
            "ag_ratio_diff_gt_0_25_count": len(ag_large_diff_indices),
            "ag_ratio_diff_gt_0_25_indices": ag_large_diff_indices,
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": {
            "gender_counts": {str(k): int(v) for k, v in gender_counts.items()},
            "gender_by_target": gender_by_target,
        },
    }
