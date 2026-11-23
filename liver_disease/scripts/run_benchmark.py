"""CLI script for Phase 4: Leakage-free train/test split and 11-model benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from liver_disease.config import RANDOM_STATE, TEST_SIZE
from liver_disease.data.loader import load_raw_dataset
from liver_disease.data.preprocessing import create_leakage_free_split
from liver_disease.models.benchmark import run_full_benchmark_suite
from liver_disease.utils.logging_config import setup_logging


def main() -> None:
    logger = setup_logging("INFO")
    df_raw = load_raw_dataset()
    X_train, X_test, y_train, y_test, split_meta = create_leakage_free_split(
        df_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        save_artifacts=True,
    )
    logger.info(
        "Split created: train=%d (pos=%d), test=%d (pos=%d), overlap=%d",
        split_meta["train_rows"],
        split_meta["train_positive_count"],
        split_meta["test_rows"],
        split_meta["test_positive_count"],
        split_meta["train_test_feature_overlap"],
    )

    # IMPORTANT: X_test and y_test are NOT passed to benchmarking!
    results = run_full_benchmark_suite(X_train=X_train, y_train=y_train, random_state=RANDOM_STATE)

    print("\n=== EXPERIMENT 1: 11-MODEL BASELINE BENCHMARK (5x3 REPEATED STRATIFIED CV) ===")
    df_base = pd.DataFrame(results["baseline_11_models"])
    cols = [
        "model",
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_score_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
    ]
    print(df_base[cols].to_string(index=False))

    print("\n=== EXPERIMENT 2: FEATURE ENGINEERING ABLATION (AGGREGATE) ===")
    print(pd.DataFrame(results["best_feature_engineering_config"]["aggregate_scores"]).T.to_string())
    print(f"Selected feature config: {results['best_feature_engineering_config']['name']}")

    print("\n=== EXPERIMENT 3: IMBALANCE STRATEGY COMPARISON (MEAN ACROSS MODELS) ===")
    df_imb = pd.DataFrame(results["imbalance_strategy_comparison"])
    print(
        df_imb.groupby("imbalance_strategy")[
            ["roc_auc_mean", "pr_auc_mean", "recall_sensitivity_mean", "specificity_mean", "balanced_accuracy_mean", "f1_score_mean"]
        ].mean().round(4).to_string()
    )

    print("\n=== EXPERIMENT 4: FEATURE SELECTION ABLATION ===")
    df_fs = pd.DataFrame(results["feature_selection_ablation"])
    print(df_fs[["model", "k_features", "roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
