"""CLI script for Phases 5 & 6: Hyperparameter tuning, calibration, final holdout evaluation,
SHAP explainability, and versioned model bundle serialization.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from liver_disease.config import REPORTS_DIR
from liver_disease.models.train import run_training_and_final_evaluation
from liver_disease.utils.io import load_json
from liver_disease.utils.logging_config import setup_logging


def main() -> None:
    logger = setup_logging("INFO")

    # Read winning feature engineering decision from Phase 4 benchmark if available
    add_ratios = False
    do_log = True
    bench_path = REPORTS_DIR / "benchmark_summary.json"
    if bench_path.exists():
        bench = load_json(bench_path)
        fe_cfg = bench.get("best_feature_engineering_config", {})
        add_ratios = bool(fe_cfg.get("add_clinical_ratios", False))
        do_log = bool(fe_cfg.get("log_transform_skewed", True))
        logger.info(
            "Loaded Phase 4 feature config: %s (add_ratios=%s, log1p=%s)",
            fe_cfg.get("name"),
            add_ratios,
            do_log,
        )

    report = run_training_and_final_evaluation(
        add_clinical_ratios=add_ratios,
        log_transform_skewed=do_log,
    )

    meta = report["model_metadata"]
    ev = meta["evaluation_metrics"]
    m_def = ev["holdout_test_default_threshold_0_50"]
    m_sel = ev["holdout_test_selected_threshold"]
    cis = ev["holdout_test_bootstrap_95_ci"]

    print("\n=== FINAL MODEL SUMMARY ===")
    print(f"Selected Model     : {meta['model_name']} ({meta['model_type']})")
    print(f"Calibration        : {meta['calibration_method']}")
    print(f"Default Threshold  : {meta['default_threshold']:.2f}")
    print(f"Selected Threshold : {meta['decision_threshold']:.2f} (Validation OOF)")
    print(f"Explainer          : {meta['explainer_type']}")

    print("\n--- HOLDOUT TEST METRICS (Default Threshold = 0.50) ---")
    print(
        f"ROC-AUC={m_def['roc_auc']:.4f} | PR-AUC={m_def['pr_auc']:.4f} | "
        f"Acc={m_def['accuracy']:.4f} | BalAcc={m_def['balanced_accuracy']:.4f} | "
        f"Prec={m_def['precision']:.4f} | Recall={m_def['recall_sensitivity']:.4f} | "
        f"Spec={m_def['specificity']:.4f} | F1={m_def['f1_score']:.4f} | "
        f"CM={m_def['confusion_matrix']}"
    )

    print(f"\n--- HOLDOUT TEST METRICS (Validation-Selected Threshold = {meta['decision_threshold']:.2f}) ---")
    print(
        f"ROC-AUC={m_sel['roc_auc']:.4f} (95% CI: {cis['roc_auc']['ci_lower']:.3f}–{cis['roc_auc']['ci_upper']:.3f}) | "
        f"PR-AUC={m_sel['pr_auc']:.4f} (95% CI: {cis['pr_auc']['ci_lower']:.3f}–{cis['pr_auc']['ci_upper']:.3f})"
    )
    print(
        f"Acc={m_sel['accuracy']:.4f} | BalAcc={m_sel['balanced_accuracy']:.4f} | "
        f"Prec={m_sel['precision']:.4f} | Recall={m_sel['recall_sensitivity']:.4f} | "
        f"Spec={m_sel['specificity']:.4f} | F1={m_sel['f1_score']:.4f} | "
        f"Brier={m_sel['brier_score']:.4f} | CM={m_sel['confusion_matrix']}"
    )

    print("\n--- TOP GLOBAL SHAP FEATURES ---")
    for item in meta["global_feature_importance"][:8]:
        print(f"  #{item['rank']}: {item['feature']:22s} mean(|SHAP|) = {item['mean_abs_shap']:.5f}")


if __name__ == "__main__":
    main()
