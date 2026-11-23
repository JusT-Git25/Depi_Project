"""Phases 5 & 6: Hyperparameter tuning, ensemble & calibration study, threshold optimization,
single-pass holdout test evaluation, SHAP explainability, and versioned artifact serialization.
"""

from __future__ import annotations

import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from liver_disease.config import (
    DEFAULT_MODEL_ARTIFACT_PATH,
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    MODEL_VERSION,
    PREPROCESSING_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    REPORTS_DIR,
    RISK_THRESHOLDS,
    ensure_project_directories,
)
from liver_disease.data.loader import get_dataset_provenance, load_raw_dataset, split_features_and_target
from liver_disease.data.preprocessing import (
    build_training_pipeline,
    create_leakage_free_split,
    get_transformed_feature_names,
)
from liver_disease.models.benchmark import evaluate_pipeline_repeated_cv
from liver_disease.models.evaluate import (
    compute_bootstrap_confidence_intervals,
    compute_classification_metrics,
    optimize_validation_threshold,
    save_evaluation_plots,
)
from liver_disease.models.explain import generate_global_shap_artifacts
from liver_disease.utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def _collect_dependency_versions() -> Dict[str, str]:
    """Capture exact runtime and package versions for strict reproducibility."""
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "shap",
        "joblib",
        "fastapi",
        "pydantic",
    ]
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in packages:
        try:
            versions[pkg] = pkg_version(pkg)
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def _load_locked_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Load the deterministic train/test split from data/processed/ (or create if missing)."""
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    meta_json = PROCESSED_DATA_DIR / "split_metadata.json"

    if train_csv.exists() and test_csv.exists() and meta_json.exists():
        X_train, y_train = split_features_and_target(pd.read_csv(train_csv))
        X_test, y_test = split_features_and_target(pd.read_csv(test_csv))
        split_meta = load_json(meta_json)
        return X_train, X_test, y_train, y_test, split_meta

    df_raw = load_raw_dataset()
    return create_leakage_free_split(df_raw, save_artifacts=True)


def run_training_and_final_evaluation(
    add_clinical_ratios: bool = False,
    log_transform_skewed: bool = True,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute Phases 5 & 6 end-to-end without leaking holdout test data."""
    ensure_project_directories()
    X_train, X_test, y_train, y_test, split_meta = _load_locked_split()
    provenance = get_dataset_provenance()

    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ------------------------------------------------------------------
    # 1. Hyperparameter Optimization on Top 4 Diverse Model Families (X_train ONLY)
    # ------------------------------------------------------------------
    logger.info("Phase 5.1: Hyperparameter optimization on X_train (n=%d)...", len(X_train))

    search_specs: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {
        "ExtraTrees_Tuned": (
            build_training_pipeline(
                estimator=ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
                scaler="none",
                add_clinical_ratios=add_clinical_ratios,
                log_transform_skewed=log_transform_skewed,
                imbalance_strategy="none",
                random_state=random_state,
            ),
            {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [6, 10, 14, None],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.6],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "RandomForest_Tuned": (
            build_training_pipeline(
                estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                scaler="none",
                add_clinical_ratios=add_clinical_ratios,
                log_transform_skewed=log_transform_skewed,
                imbalance_strategy="none",
                random_state=random_state,
            ),
            {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [5, 8, 12, None],
                "classifier__min_samples_leaf": [2, 4, 6],
                "classifier__max_features": ["sqrt", 0.5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "LogisticRegression_Tuned": (
            build_training_pipeline(
                estimator=LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state),
                scaler="robust",
                add_clinical_ratios=add_clinical_ratios,
                log_transform_skewed=log_transform_skewed,
                imbalance_strategy="none",
                random_state=random_state,
            ),
            {
                "classifier__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "CatBoost_Tuned": (
            build_training_pipeline(
                estimator=CatBoostClassifier(
                    random_seed=random_state,
                    verbose=0,
                    allow_writing_files=False,
                ),
                scaler="none",
                add_clinical_ratios=add_clinical_ratios,
                log_transform_skewed=log_transform_skewed,
                imbalance_strategy="none",
                random_state=random_state,
            ),
            {
                "classifier__iterations": [200, 350],
                "classifier__learning_rate": [0.02, 0.04],
                "classifier__depth": [3, 4, 5],
                "classifier__l2_leaf_reg": [3.0, 7.0],
                "classifier__auto_class_weights": [None, "Balanced"],
            },
        ),
    }

    tuned_estimators: Dict[str, Any] = {}
    tuned_cv_results: List[Dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (pipe, param_grid) in search_specs.items():
            logger.info("  Tuning %s...", name)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
                refit="roc_auc",
                cv=cv_tune,
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            tuned_estimators[name] = best_pipe

            # Evaluate consistently with 5x3 Repeated Stratified CV
            rep_cv = evaluate_pipeline_repeated_cv(
                pipeline_builder=lambda bp=best_pipe: clone(bp),
                X_train=X_train,
                y_train=y_train,
                n_splits=5,
                n_repeats=3,
                random_state=random_state,
            )
            tuned_cv_results.append(
                {
                    "candidate": name,
                    "best_params": {k.replace("classifier__", ""): v for k, v in gs.best_params_.items()},
                    "grid_best_roc_auc": round(float(gs.best_score_), 4),
                    **rep_cv,
                }
            )
            logger.info(
                "    %s -> Repeated-CV ROC-AUC=%.4f±%.4f, PR-AUC=%.4f±%.4f, Brier=%.4f",
                name,
                rep_cv["roc_auc_mean"],
                rep_cv["roc_auc_std"],
                rep_cv["pr_auc_mean"],
                rep_cv["pr_auc_std"],
                rep_cv["brier_score_mean"],
            )

    # ------------------------------------------------------------------
    # 2. Soft-Voting Ensemble Experiment (ExtraTrees + RandomForest + LogisticRegression)
    # ------------------------------------------------------------------
    logger.info("Phase 5.2: Evaluating Soft-Voting Ensemble of top diverse tuned models...")
    voting_estimator = VotingClassifier(
        estimators=[
            ("et", clone(tuned_estimators["ExtraTrees_Tuned"].named_steps["classifier"])),
            ("rf", clone(tuned_estimators["RandomForest_Tuned"].named_steps["classifier"])),
            ("lr", clone(tuned_estimators["LogisticRegression_Tuned"].named_steps["classifier"])),
        ],
        voting="soft",
        weights=[2, 1, 1],
    )
    voting_pipe = build_training_pipeline(
        estimator=voting_estimator,
        scaler="robust",
        add_clinical_ratios=add_clinical_ratios,
        log_transform_skewed=log_transform_skewed,
        imbalance_strategy="none",
        random_state=random_state,
    )
    voting_cv = evaluate_pipeline_repeated_cv(
        pipeline_builder=lambda vp=voting_pipe: clone(vp),
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        n_repeats=3,
        random_state=random_state,
    )
    tuned_estimators["SoftVoting_ET_RF_LR"] = voting_pipe
    tuned_cv_results.append(
        {
            "candidate": "SoftVoting_ET_RF_LR",
            "best_params": {"members": ["ExtraTrees_Tuned", "RandomForest_Tuned", "LogisticRegression_Tuned"], "weights": [2, 1, 1]},
            "grid_best_roc_auc": voting_cv["roc_auc_mean"],
            **voting_cv,
        }
    )
    logger.info(
        "    SoftVoting_ET_RF_LR -> Repeated-CV ROC-AUC=%.4f±%.4f, PR-AUC=%.4f±%.4f, Brier=%.4f",
        voting_cv["roc_auc_mean"],
        voting_cv["roc_auc_std"],
        voting_cv["pr_auc_mean"],
        voting_cv["pr_auc_std"],
        voting_cv["brier_score_mean"],
    )

    # Select winning candidate by pre-registered primary CV criterion:
    # Composite discrimination rank = 0.5 * ROC-AUC + 0.5 * PR-AUC (with Brier tie-breaker)
    for row in tuned_cv_results:
        row["selection_score"] = round(0.5 * row["roc_auc_mean"] + 0.5 * row["pr_auc_mean"], 4)

    tuned_df = pd.DataFrame(tuned_cv_results).sort_values(
        by=["selection_score", "roc_auc_mean", "pr_auc_mean", "brier_score_mean"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    tuned_df.to_csv(REPORTS_DIR / "tuned_candidates_cv_comparison.csv", index=False)

    winner_name = str(tuned_df.loc[0, "candidate"])
    winner_base_pipe = clone(tuned_estimators[winner_name])
    winner_cv_summary = tuned_df.loc[0].to_dict()
    logger.info("Selected winning candidate from CV: %s (selection_score=%.4f)", winner_name, winner_cv_summary["selection_score"])

    # ------------------------------------------------------------------
    # 3. Probability Calibration Experiment on Winning Model (CV ONLY)
    # ------------------------------------------------------------------
    logger.info("Phase 5.3: Evaluating probability calibration (none vs sigmoid vs isotonic) in CV...")
    cal_candidates: Dict[str, Any] = {
        "none": lambda: clone(winner_base_pipe),
        "sigmoid": lambda: CalibratedClassifierCV(
            estimator=clone(winner_base_pipe),
            method="sigmoid",
            cv=5,
        ),
        "isotonic": lambda: CalibratedClassifierCV(
            estimator=clone(winner_base_pipe),
            method="isotonic",
            cv=5,
        ),
    }

    cal_rows: List[Dict[str, Any]] = []
    for cal_name, builder in cal_candidates.items():
        m = evaluate_pipeline_repeated_cv(
            pipeline_builder=builder,
            X_train=X_train,
            y_train=y_train,
            n_splits=5,
            n_repeats=3,
            random_state=random_state,
        )
        cal_rows.append({"calibration": cal_name, **m})
        logger.info(
            "  Calibration=%-8s | Brier=%.4f±%.4f | ROC-AUC=%.4f | PR-AUC=%.4f",
            cal_name,
            m["brier_score_mean"],
            m["brier_score_std"],
            m["roc_auc_mean"],
            m["pr_auc_mean"],
        )

    cal_df = pd.DataFrame(cal_rows)
    none_row = cal_df[cal_df["calibration"] == "none"].iloc[0]
    best_cal_row = cal_df.sort_values(by=["brier_score_mean", "roc_auc_mean"], ascending=[True, False]).iloc[0]

    # Adopt calibration if it measurably lowers Brier score without losing >0.005 ROC-AUC
    if (
        best_cal_row["calibration"] != "none"
        and best_cal_row["brier_score_mean"] <= none_row["brier_score_mean"] - 0.002
        and best_cal_row["roc_auc_mean"] >= none_row["roc_auc_mean"] - 0.005
    ):
        selected_calibration = str(best_cal_row["calibration"])
    else:
        selected_calibration = "none"

    logger.info("Selected calibration method: %s", selected_calibration)
    final_model_builder = cal_candidates[selected_calibration]

    # ------------------------------------------------------------------
    # 4. Out-of-Fold Validation Threshold Optimization (NEVER TOUCHING X_test)
    # ------------------------------------------------------------------
    logger.info("Phase 5.4: Optimizing decision threshold on out-of-fold validation predictions...")
    cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_proba = cross_val_predict(
            final_model_builder(),
            X_train,
            y_train,
            cv=cv_oof,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

    selected_threshold, threshold_report = optimize_validation_threshold(
        y_true_oof=y_train.to_numpy(),
        y_proba_oof=oof_proba,
        min_specificity=0.50,
    )
    logger.info(
        "Validation threshold optimization complete: default=0.50 -> selected=%.2f (OOF Recall=%.4f, Specificity=%.4f, F1=%.4f)",
        selected_threshold,
        threshold_report["selected_oof_metrics"]["recall_sensitivity"],
        threshold_report["selected_oof_metrics"]["specificity"],
        threshold_report["selected_oof_metrics"]["f1_score"],
    )

    # ------------------------------------------------------------------
    # 5. Fit Final Pipeline on Full Training Split (X_train, y_train)
    # ------------------------------------------------------------------
    logger.info("Phase 5.5: Fitting frozen final pipeline on full X_train (n=%d)...", len(X_train))
    final_pipeline = final_model_builder()
    explain_pipeline = clone(winner_base_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X_train, y_train)
        explain_pipeline.fit(X_train, y_train)

    transformed_feature_names = get_transformed_feature_names(explain_pipeline)

    # ------------------------------------------------------------------
    # 6. SINGLE-PASS FINAL EVALUATION ON LOCKED HOLDOUT TEST SET (X_test, y_test)
    # ------------------------------------------------------------------
    logger.info("Phase 5.6: Evaluating frozen model ONCE on locked holdout test set (n=%d)...", len(X_test))
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics_default = compute_classification_metrics(y_test.to_numpy(), y_test_proba, threshold=0.50)
    test_metrics_selected = compute_classification_metrics(
        y_test.to_numpy(),
        y_test_proba,
        threshold=selected_threshold,
    )
    bootstrap_cis_selected = compute_bootstrap_confidence_intervals(
        y_true=y_test.to_numpy(),
        y_proba=y_test_proba,
        threshold=selected_threshold,
        n_bootstraps=1000,
        confidence_level=0.95,
        random_state=random_state,
    )
    eval_plot_paths = save_evaluation_plots(
        y_true=y_test.to_numpy(),
        y_proba=y_test_proba,
        threshold=selected_threshold,
        model_name=winner_name,
        figures_dir=FIGURES_DIR,
        prefix="final_test",
    )

    # ------------------------------------------------------------------
    # 7. Phase 6: SHAP Explainability & Complete Artifact Serialization
    # ------------------------------------------------------------------
    logger.info("Phase 6: Generating SHAP global importance, summary plots, and serializing inference bundle...")
    shap_artifacts = generate_global_shap_artifacts(
        pipeline=explain_pipeline,
        X_train=X_train,
        X_eval=X_test,
        feature_names=transformed_feature_names,
        figures_dir=FIGURES_DIR,
        max_background=120,
    )

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    dep_versions = _collect_dependency_versions()
    model_type = type(explain_pipeline.named_steps["classifier"]).__name__

    evaluation_summary = {
        "cv_winner_repeated_5x3": winner_cv_summary,
        "calibration_comparison": cal_rows,
        "selected_calibration": selected_calibration,
        "oof_threshold_optimization": {
            k: v for k, v in threshold_report.items() if k != "threshold_sweep"
        },
        "holdout_test_default_threshold_0_50": test_metrics_default,
        "holdout_test_selected_threshold": test_metrics_selected,
        "holdout_test_bootstrap_95_ci": bootstrap_cis_selected,
    }

    training_config = {
        "random_state": random_state,
        "add_clinical_ratios": add_clinical_ratios,
        "log_transform_skewed": log_transform_skewed,
        "selected_candidate": winner_name,
        "best_params": winner_cv_summary.get("best_params", {}),
        "calibration_method": selected_calibration,
        "default_threshold": 0.50,
        "decision_threshold": selected_threshold,
        "min_specificity_guardrail": 0.50,
    }

    dataset_info = {
        "provenance": provenance,
        "split_metadata": {
            "rows_raw": split_meta["deduplication"]["rows_before"],
            "exact_duplicates_removed": split_meta["deduplication"]["exact_duplicates_removed"],
            "rows_deduplicated": split_meta["deduplication"]["rows_after"],
            "train_rows": split_meta["train_rows"],
            "test_rows": split_meta["test_rows"],
            "train_positive_rate": split_meta["train_positive_rate"],
            "test_positive_rate": split_meta["test_positive_rate"],
            "train_test_feature_overlap": split_meta["train_test_feature_overlap"],
        },
    }

    artifact_bundle = {
        "artifact_format_version": "1.0",
        "model_version": MODEL_VERSION,
        "model_name": winner_name,
        "model_type": model_type,
        "pipeline": final_pipeline,
        "explain_pipeline": explain_pipeline,
        "calibration_method": selected_calibration,
        "default_threshold": 0.50,
        "decision_threshold": selected_threshold,
        "risk_thresholds": RISK_THRESHOLDS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "transformed_feature_names": transformed_feature_names,
        "shap_background": shap_artifacts["shap_background_sample"],
        "global_feature_importance": shap_artifacts["global_feature_importance"],
        "training_config": training_config,
        "dataset_info": dataset_info,
        "evaluation_metrics": evaluation_summary,
        "trained_at_utc": trained_at_utc,
        "dependency_versions": dep_versions,
    }

    joblib.dump(artifact_bundle, DEFAULT_MODEL_ARTIFACT_PATH, compress=3)
    logger.info("Saved serialized inference pipeline bundle to %s", DEFAULT_MODEL_ARTIFACT_PATH)

    # Save JSON metadata & preprocessing schema snapshots
    metadata_json = {
        k: v
        for k, v in artifact_bundle.items()
        if k not in {"pipeline", "explain_pipeline", "shap_background"}
    }
    metadata_json["artifact_path"] = str(DEFAULT_MODEL_ARTIFACT_PATH)
    metadata_json["artifact_size_bytes"] = DEFAULT_MODEL_ARTIFACT_PATH.stat().st_size
    metadata_json["explainer_type"] = shap_artifacts["explainer_type"]
    metadata_json["figures"] = {**eval_plot_paths, **shap_artifacts["figures"]}
    save_json(metadata_json, MODEL_METADATA_PATH)

    # Save preprocessing metadata snapshot in artifacts/preprocessing/
    preprocessor_snapshot = {
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "transformed_feature_names": transformed_feature_names,
        "add_clinical_ratios": add_clinical_ratios,
        "log_transform_skewed": log_transform_skewed,
        "numeric_imputation": "median (fitted strictly on X_train)",
        "categorical_encoding": "OneHotEncoder(Female/Male, drop=if_binary)",
        "scaler": "robust" if "Logistic" in winner_name or "Voting" in winner_name else "none",
    }
    save_json(preprocessor_snapshot, PREPROCESSING_DIR / "preprocessing_schema.json")

    full_report = {
        "model_metadata": metadata_json,
        "tuned_candidates_comparison": tuned_cv_results,
        "threshold_optimization_full": threshold_report,
    }
    save_json(full_report, REPORTS_DIR / "final_evaluation_report.json")
    logger.info("Saved final evaluation report to %s", REPORTS_DIR / "final_evaluation_report.json")
    return full_report
