"""Phase 4: Comprehensive 11-model benchmarking, imbalance, feature engineering, and selection studies."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from liver_disease.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, ensure_project_directories
from liver_disease.data.preprocessing import (
    ImbalanceStrategy,
    ScalerType,
    build_training_pipeline,
)
from liver_disease.models.evaluate import compute_classification_metrics
from liver_disease.utils.io import save_json

logger = logging.getLogger(__name__)


def get_candidate_models(
    use_class_weight: bool = False,
    pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Tuple[Any, ScalerType]]:
    """Return all 11 required candidate classifiers paired with their appropriate scaler.

    Note: In ILPD, Class 1 (Liver Disease) is ~71.2% and Class 0 (No Liver Disease) is ~28.8%.
    When `use_class_weight=True`, `class_weight="balanced"` (or equivalent sample weight ratio)
    rebalances both classes equally so models do not collapse onto predicting the majority class.
    """
    cw = "balanced" if use_class_weight else None
    xgb_spw = pos_weight if use_class_weight else 1.0

    return {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=2000,
                class_weight=cw,
                random_state=random_state,
                solver="lbfgs",
            ),
            "robust",
        ),
        "DecisionTree": (
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=5,
                class_weight=cw,
                random_state=random_state,
            ),
            "none",
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                class_weight=cw,
                random_state=random_state,
                n_jobs=-1,
            ),
            "none",
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight=cw,
                random_state=random_state,
                n_jobs=-1,
            ),
            "none",
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.85,
                random_state=random_state,
            ),
            "none",
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=xgb_spw,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ),
            "none",
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight=cw,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
            "none",
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=300,
                learning_rate=0.04,
                depth=4,
                auto_class_weights="Balanced" if use_class_weight else None,
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            ),
            "none",
        ),
        "SVM_RBF": (
            SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight=cw,
                random_state=random_state,
            ),
            "robust",
        ),
        "KNN": (
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
            ),
            "robust",
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=15,
                class_weight=cw,
                random_state=random_state,
            ),
            "none",
        ),
    }


def evaluate_pipeline_repeated_cv(
    pipeline_builder: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Evaluate a pipeline with Repeated Stratified K-Fold CV (zero test-set usage)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in rskf.split(X_reset, y_reset):
            pipe = clone(pipeline_builder())
            pipe.fit(X_reset.iloc[train_idx], y_reset.iloc[train_idx])
            y_val_proba = pipe.predict_proba(X_reset.iloc[val_idx])[:, 1]
            m = compute_classification_metrics(y_reset.iloc[val_idx].to_numpy(), y_val_proba, threshold=0.50)
            fold_metrics.append(
                {
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "precision": m["precision"],
                    "recall_sensitivity": m["recall_sensitivity"],
                    "specificity": m["specificity"],
                    "f1_score": m["f1_score"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "brier_score": m["brier_score"],
                }
            )

    df_folds = pd.DataFrame(fold_metrics)
    summary: Dict[str, Any] = {}
    for col in df_folds.columns:
        summary[f"{col}_mean"] = round(float(df_folds[col].mean()), 4)
        summary[f"{col}_std"] = round(float(df_folds[col].std()), 4)
    return summary


def run_full_benchmark_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Run all Phase 4 benchmarking experiments strictly on X_train / y_train."""
    ensure_project_directories()

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)  # Ratio for XGBoost balanced weighting (~0.40)

    # ------------------------------------------------------------------
    # Experiment 1: 11-Model Baseline Comparison (5x3 Repeated Stratified CV)
    # ------------------------------------------------------------------
    logger.info("Running Experiment 1: 11-Model Baseline Benchmark (5x3 Repeated Stratified CV)...")
    baseline_rows: List[Dict[str, Any]] = []
    candidates = get_candidate_models(use_class_weight=False, pos_weight=pos_weight, random_state=random_state)

    for model_name, (estimator, scaler) in candidates.items():
        logger.info("  Evaluating %s (baseline)...", model_name)
        metrics = evaluate_pipeline_repeated_cv(
            pipeline_builder=lambda est=estimator, sc=scaler: build_training_pipeline(
                estimator=est,
                scaler=sc,
                add_clinical_ratios=False,
                log_transform_skewed=True,
                imbalance_strategy="none",
                random_state=random_state,
            ),
            X_train=X_train,
            y_train=y_train,
            n_splits=5,
            n_repeats=3,
            random_state=random_state,
        )
        row = {"model": model_name, "scaler": scaler, **metrics}
        baseline_rows.append(row)

    baseline_df = pd.DataFrame(baseline_rows).sort_values(
        by=["roc_auc_mean", "pr_auc_mean"], ascending=False
    ).reset_index(drop=True)
    baseline_df.to_csv(REPORTS_DIR / "benchmark_11_models_baseline.csv", index=False)

    # ------------------------------------------------------------------
    # Experiment 2: Feature Engineering Ablation
    # Compare Raw vs Log1p vs Log1p + Clinical Ratios across 5 diverse families
    # ------------------------------------------------------------------
    logger.info("Running Experiment 2: Feature Engineering Ablation...")
    fe_models = ["LogisticRegression", "RandomForest", "ExtraTrees", "CatBoost", "SVM_RBF"]
    fe_configs = [
        ("raw_only", False, False),
        ("log1p_only", False, True),
        ("log1p_plus_clinical_ratios", True, True),
    ]
    fe_rows: List[Dict[str, Any]] = []
    for cfg_name, add_ratios, do_log in fe_configs:
        for model_name in fe_models:
            est, scaler = candidates[model_name]
            metrics = evaluate_pipeline_repeated_cv(
                pipeline_builder=lambda e=est, s=scaler, ar=add_ratios, dl=do_log: build_training_pipeline(
                    estimator=e,
                    scaler=s,
                    add_clinical_ratios=ar,
                    log_transform_skewed=dl,
                    imbalance_strategy="none",
                    random_state=random_state,
                ),
                X_train=X_train,
                y_train=y_train,
                n_splits=5,
                n_repeats=3,
                random_state=random_state,
            )
            fe_rows.append({"feature_set": cfg_name, "model": model_name, **metrics})

    fe_df = pd.DataFrame(fe_rows)
    fe_df.to_csv(REPORTS_DIR / "benchmark_feature_engineering_ablation.csv", index=False)

    # Pick winning feature configuration by mean ROC-AUC + PR-AUC across models
    fe_agg = fe_df.groupby("feature_set")[["roc_auc_mean", "pr_auc_mean"]].mean()
    best_fe_name = str(fe_agg.mean(axis=1).idxmax())
    fe_map = {name: (ar, dl) for name, ar, dl in fe_configs}
    best_add_ratios, best_do_log = fe_map[best_fe_name]
    logger.info("Selected best feature configuration from CV: %s", best_fe_name)

    # ------------------------------------------------------------------
    # Experiment 3: Class Imbalance Strategy Comparison (In-Fold Only)
    # none vs class_weight vs RandomOverSampler vs SMOTE
    # ------------------------------------------------------------------
    logger.info("Running Experiment 3: Class Imbalance Strategy Comparison (in-fold only)...")
    imb_models = ["LogisticRegression", "RandomForest", "ExtraTrees", "LightGBM", "CatBoost", "XGBoost"]
    strategies: List[ImbalanceStrategy] = ["none", "class_weight", "random_oversampler", "smote"]
    imb_rows: List[Dict[str, Any]] = []

    weighted_candidates = get_candidate_models(use_class_weight=True, pos_weight=pos_weight, random_state=random_state)
    for strat in strategies:
        for model_name in imb_models:
            if strat == "class_weight":
                est, scaler = weighted_candidates[model_name]
                pipe_strat: ImbalanceStrategy = "none"
            else:
                est, scaler = candidates[model_name]
                pipe_strat = strat

            metrics = evaluate_pipeline_repeated_cv(
                pipeline_builder=lambda e=est, s=scaler, ps=pipe_strat: build_training_pipeline(
                    estimator=e,
                    scaler=s,
                    add_clinical_ratios=best_add_ratios,
                    log_transform_skewed=best_do_log,
                    imbalance_strategy=ps,
                    random_state=random_state,
                ),
                X_train=X_train,
                y_train=y_train,
                n_splits=5,
                n_repeats=3,
                random_state=random_state,
            )
            imb_rows.append({"imbalance_strategy": strat, "model": model_name, **metrics})

    imb_df = pd.DataFrame(imb_rows)
    imb_df.to_csv(REPORTS_DIR / "benchmark_imbalance_strategies.csv", index=False)

    # ------------------------------------------------------------------
    # Experiment 4: Feature Selection Ablation (SelectKBest MI inside CV)
    # ------------------------------------------------------------------
    logger.info("Running Experiment 4: Feature Selection Ablation...")
    fs_rows: List[Dict[str, Any]] = []
    k_values = [6, 8, 10, "all"]
    for model_name in ["LogisticRegression", "RandomForest", "CatBoost"]:
        est, scaler = candidates[model_name]
        for k in k_values:
            def _make_fs_pipe(e=est, s=scaler, kval=k):
                base_pipe = build_training_pipeline(
                    estimator=e,
                    scaler=s,
                    add_clinical_ratios=best_add_ratios,
                    log_transform_skewed=best_do_log,
                    imbalance_strategy="none",
                    random_state=random_state,
                )
                if kval == "all":
                    return base_pipe
                return Pipeline(
                    steps=[
                        ("preprocessor", base_pipe.named_steps["preprocessor"]),
                        (
                            "selector",
                            SelectKBest(
                                score_func=lambda Xm, ym: mutual_info_classif(Xm, ym, random_state=random_state),
                                k=kval,
                            ),
                        ),
                        ("classifier", e),
                    ]
                )

            metrics = evaluate_pipeline_repeated_cv(
                pipeline_builder=_make_fs_pipe,
                X_train=X_train,
                y_train=y_train,
                n_splits=5,
                n_repeats=3,
                random_state=random_state,
            )
            fs_rows.append({"model": model_name, "k_features": str(k), **metrics})

    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(REPORTS_DIR / "benchmark_feature_selection.csv", index=False)

    # Plot 11-model comparison chart
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    plot_df = baseline_df.sort_values("roc_auc_mean", ascending=True)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.18, plot_df["roc_auc_mean"], height=0.34, xerr=plot_df["roc_auc_std"], label="ROC-AUC", color="#2C7BB6", alpha=0.9)
    ax.barh(y_pos + 0.18, plot_df["pr_auc_mean"], height=0.34, xerr=plot_df["pr_auc_std"], label="PR-AUC", color="#D7191C", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("Cross-Validation Score (Mean ± SD, 5×3 Repeated Stratified CV)")
    ax.set_title("11-Algorithm Benchmark on ILPD Training Split", fontweight="bold")
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_model_benchmark_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    results = {
        "cv_protocol": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)",
        "train_rows": int(len(X_train)),
        "train_positive_count": n_pos,
        "train_negative_count": n_neg,
        "best_feature_engineering_config": {
            "name": best_fe_name,
            "add_clinical_ratios": best_add_ratios,
            "log_transform_skewed": best_do_log,
            "aggregate_scores": fe_agg.round(4).to_dict(orient="index"),
        },
        "baseline_11_models": baseline_df.to_dict(orient="records"),
        "feature_engineering_ablation": fe_df.to_dict(orient="records"),
        "imbalance_strategy_comparison": imb_df.to_dict(orient="records"),
        "feature_selection_ablation": fs_df.to_dict(orient="records"),
    }
    save_json(results, REPORTS_DIR / "benchmark_summary.json")
    logger.info("Saved full benchmark results to %s", REPORTS_DIR / "benchmark_summary.json")
    return results
