"""SHAP-based global and local model explainability for the Liver Disease Prediction System.

Important methodological note:
SHAP values quantify how each input feature shifts a specific model's prediction
relative to its training-set baseline. They reflect statistical associations learned
by the model on ILPD — they MUST NOT be interpreted as evidence of medical causation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from liver_disease.config import MEDICAL_DISCLAIMER, RANDOM_STATE

logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)

ATTRIBUTION_LIMITATIONS_NOTE: str = (
    "Feature attributions (SHAP values) quantify each feature's statistical contribution "
    "to this model's output relative to the ILPD training baseline. Correlated biomarkers "
    "(e.g., Total vs Direct Bilirubin, AST vs ALT) share attribution weight, and feature "
    "importance is NOT evidence of biological causation or clinical diagnosis."
)


def _extract_preprocessor_and_estimator(pipeline: Any) -> Tuple[Any, Any]:
    """Extract the fitted preprocessor and final classifier from a sklearn or imblearn Pipeline."""
    if hasattr(pipeline, "named_steps"):
        steps = pipeline.named_steps
        classifier = steps["classifier"]
        if "preprocessor" in steps:
            return steps["preprocessor"], classifier
        if "feature_engineer" in steps and "column_transformer" in steps:
            from sklearn.pipeline import Pipeline

            pre = Pipeline(
                steps=[
                    ("feature_engineer", steps["feature_engineer"]),
                    ("column_transformer", steps["column_transformer"]),
                ]
            )
            return pre, classifier
    raise ValueError(
        "Expected a fitted scikit-learn/imblearn Pipeline with 'preprocessor' (or "
        "'feature_engineer' + 'column_transformer') and 'classifier' steps."
    )


def _extract_positive_class_shap(raw_values: Any) -> np.ndarray:
    """Normalize SHAP output across SHAP versions and binary/multiclass formats to 2D (n_samples, n_features)."""
    if hasattr(raw_values, "values"):
        arr = np.asarray(raw_values.values, dtype=float)
    elif isinstance(raw_values, list):
        # Older binary tree/kernel explainers return [class0_array, class1_array]
        arr = np.asarray(raw_values[1] if len(raw_values) == 2 else raw_values[0], dtype=float)
    else:
        arr = np.asarray(raw_values, dtype=float)

    if arr.ndim == 3:
        # Shape (n_samples, n_features, 2) -> positive class index 1
        arr = arr[:, :, 1]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _extract_positive_base_value(expected_value: Any) -> float:
    """Extract scalar expected value for positive class (index 1)."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value, dtype=float).ravel()
        return float(arr[1] if len(arr) >= 2 else arr[0])
    return float(expected_value)


def _is_tree_estimator(estimator: Any) -> bool:
    """Check whether estimator is natively supported by shap.TreeExplainer."""
    cls_name = type(estimator).__name__
    return cls_name in {
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
    }


def _is_linear_estimator(estimator: Any) -> bool:
    """Check whether estimator is natively supported by shap.LinearExplainer."""
    cls_name = type(estimator).__name__
    return cls_name in {"LogisticRegression", "SGDClassifier", "RidgeClassifier"}


def build_shap_explainer(
    pipeline: Any,
    X_background_transformed: np.ndarray,
) -> Tuple[Any, str]:
    """Create the most accurate & fast SHAP explainer appropriate for the classifier."""
    _, estimator = _extract_preprocessor_and_estimator(pipeline)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _is_tree_estimator(estimator):
            try:
                explainer = shap.TreeExplainer(estimator)
                return explainer, "TreeExplainer"
            except Exception as exc:
                logger.warning("TreeExplainer failed (%s); falling back to KernelExplainer.", exc)

        if _is_linear_estimator(estimator):
            try:
                explainer = shap.LinearExplainer(estimator, X_background_transformed)
                return explainer, "LinearExplainer"
            except Exception as exc:
                logger.warning("LinearExplainer failed (%s); falling back to KernelExplainer.", exc)

        # Model-agnostic fallback (works for VotingClassifier, CalibratedClassifierCV, SVM, KNN, etc.)
        bg_summary = shap.kmeans(X_background_transformed, min(30, len(X_background_transformed)))
        explainer = shap.KernelExplainer(lambda x: estimator.predict_proba(x)[:, 1], bg_summary)
        return explainer, "KernelExplainer"


def compute_shap_values(
    pipeline: Any,
    X_raw: pd.DataFrame,
    X_background_transformed: np.ndarray,
    explainer: Any | None = None,
    explainer_type: str | None = None,
) -> Tuple[np.ndarray, float, np.ndarray, str]:
    """Compute SHAP values for X_raw using the pipeline's preprocessor and classifier."""
    preprocessor, _ = _extract_preprocessor_and_estimator(pipeline)
    X_trans = np.asarray(preprocessor.transform(X_raw), dtype=float)

    if explainer is None:
        explainer, explainer_type = build_shap_explainer(pipeline, X_background_transformed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if explainer_type == "KernelExplainer":
            raw_sv = explainer.shap_values(X_trans, nsamples=100, silent=True)
        else:
            raw_sv = explainer.shap_values(X_trans)

    sv = _extract_positive_class_shap(raw_sv)
    base_val = _extract_positive_base_value(explainer.expected_value)
    return sv, base_val, X_trans, str(explainer_type or type(explainer).__name__)


def generate_global_shap_artifacts(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    feature_names: List[str],
    figures_dir: Path,
    max_background: int = 120,
) -> Dict[str, Any]:
    """Compute global SHAP importance and save summary beeswarm + bar plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    preprocessor, _ = _extract_preprocessor_and_estimator(pipeline)

    X_bg_df = X_train.sample(n=min(max_background, len(X_train)), random_state=RANDOM_STATE)
    X_bg_trans = np.asarray(preprocessor.transform(X_bg_df), dtype=float)

    explainer, explainer_type = build_shap_explainer(pipeline, X_bg_trans)
    shap_vals, base_val, X_eval_trans, explainer_type = compute_shap_values(
        pipeline=pipeline,
        X_raw=X_eval,
        X_background_transformed=X_bg_trans,
        explainer=explainer,
        explainer_type=explainer_type,
    )

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    ranking_idx = np.argsort(mean_abs)[::-1]
    global_importance = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": round(float(mean_abs[idx]), 5),
        }
        for rank, idx in enumerate(ranking_idx)
    ]

    # 1. Global SHAP summary beeswarm plot
    plt.figure(figsize=(10, 6.2))
    shap.summary_plot(
        shap_vals,
        features=X_eval_trans,
        feature_names=feature_names,
        show=False,
        max_display=len(feature_names),
    )
    plt.title("SHAP Summary Plot (Global Feature Impact on Liver Disease Score)", fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = figures_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    # 2. Global SHAP mean |SHAP| bar plot
    plt.figure(figsize=(9.5, 5.5))
    top_names = [item["feature"] for item in global_importance][::-1]
    top_vals = [item["mean_abs_shap"] for item in global_importance][::-1]
    plt.barh(top_names, top_vals, color="#2C7BB6", edgecolor="black", alpha=0.88)
    plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)")
    plt.title("Global Feature Importance (SHAP)", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = figures_dir / "shap_global_importance_bar.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "explainer_type": explainer_type,
        "base_value": round(float(base_val), 5),
        "global_feature_importance": global_importance,
        "shap_background_sample": X_bg_trans,
        "figures": {
            "beeswarm": str(beeswarm_path),
            "bar": str(bar_path),
        },
        "limitations_note": ATTRIBUTION_LIMITATIONS_NOTE,
    }


def explain_single_record(
    pipeline: Any,
    record_df: pd.DataFrame,
    feature_names: List[str],
    X_background_transformed: np.ndarray,
    explainer: Any | None = None,
    explainer_type: str | None = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Compute individual patient SHAP explanation for a single-row DataFrame."""
    if len(record_df) != 1:
        raise ValueError("explain_single_record expects a DataFrame with exactly 1 row.")

    shap_vals, base_val, X_trans, used_type = compute_shap_values(
        pipeline=pipeline,
        X_raw=record_df,
        X_background_transformed=X_background_transformed,
        explainer=explainer,
        explainer_type=explainer_type,
    )
    row_sv = shap_vals[0]
    row_trans = X_trans[0]

    # Map original raw values when available
    raw_row = record_df.iloc[0].to_dict()
    contributions: List[Dict[str, Any]] = []
    for feat_name, sv_val, trans_val in zip(feature_names, row_sv, row_trans):
        raw_val: Any = raw_row.get(feat_name)
        if feat_name.startswith("Gender"):
            raw_val = raw_row.get("Gender")
        contributions.append(
            {
                "feature": feat_name,
                "raw_value": raw_val if raw_val is not None else round(float(trans_val), 4),
                "transformed_value": round(float(trans_val), 4),
                "shap_value": round(float(sv_val), 5),
                "abs_impact": round(float(abs(sv_val)), 5),
                "direction": "increases_risk" if sv_val >= 0 else "decreases_risk",
            }
        )

    contributions.sort(key=lambda item: item["abs_impact"], reverse=True)
    return {
        "explainer_type": used_type,
        "base_value": round(float(base_val), 5),
        "top_contributing_features": contributions[:top_k],
        "all_feature_contributions": contributions,
        "limitations_note": ATTRIBUTION_LIMITATIONS_NOTE,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
