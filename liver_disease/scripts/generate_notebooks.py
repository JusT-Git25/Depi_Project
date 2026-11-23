"""Generate and execute all 5 structured experimentation Jupyter notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _make_nb(cells: List[Tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    return nb


def build_notebooks() -> List[Path]:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    setup_snippet = (
        "import sys\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(PROJECT_ROOT / 'src') not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "from IPython.display import display, Image\n"
    )

    notebooks: dict[str, List[Tuple[str, str]]] = {
        "01_data_audit.ipynb": [
            ("md", "# 01 — ILPD Data Audit & Provenance Verification\n\nThis notebook verifies the original **Indian Liver Patient Dataset (ILPD)** structure, SHA-256 provenance, missing values, exact & near duplicates, domain consistency checks, and outlier profiles using reusable modules from `src/liver_disease/data/`."),
            ("code", setup_snippet + "from liver_disease.data.loader import ensure_raw_dataset, get_dataset_provenance, load_raw_dataset\nfrom liver_disease.data.validation import run_comprehensive_data_audit\n\nraw_path = ensure_raw_dataset()\nprov = get_dataset_provenance(raw_path)\ndf = load_raw_dataset(raw_path)\nprint('Dataset Provenance:')\nfor k, v in prov.items():\n    print(f'  {k}: {v}')\ndisplay(df.head())"),
            ("code", "audit = run_comprehensive_data_audit(df)\nprint('Shape:', audit['shape'])\nprint('Target distribution:', audit['target_distribution'])\nprint('Missing values:', audit['missing_values'])\nprint('Exact duplicate extra rows:', audit['duplicates']['exact_duplicate_extra_rows'])\nprint('Target conflicts among feature duplicates:', len(audit['duplicates']['feature_duplicate_target_conflicts']))\nprint('Near-duplicate pairs (std L2 < 0.25):', len(audit['duplicates']['near_duplicate_pairs_z_lt_0_25']))\nprint('Domain consistency checks:', audit['domain_consistency_checks'])"),
            ("code", "num_summary = pd.DataFrame(audit['numeric_summary']).T\ndisplay(num_summary[['count', 'missing', 'mean', 'std', 'min', 'median', 'max', 'skewness', 'iqr_outlier_count', 'iqr_outlier_pct']])"),
            ("md", "## Key Audit Decisions\n\n1. **Missing values**: Exactly 4 missing entries in `A/G Ratio` (0.69%). Imputed strictly inside training folds via median imputation.\n2. **Exact duplicates**: 13 exact duplicate rows (all consecutive adjacent double-entries, 0 target conflicts). Deduplicated (`583 -> 570`) before train/test splitting to prevent identical-patient leakage.\n3. **Outliers & anomalies**: Extreme bilirubin/enzyme values (11–14% IQR outliers) represent genuine severe liver pathology and are preserved intact. The 3 rows with `DB > TB` are documented transparently without silent modification."),
        ],
        "02_eda.ipynb": [
            ("md", "# 02 — Exploratory Data Analysis (EDA)\n\nStatistical distribution analysis, Mann-Whitney U tests, Spearman rank correlations, and hepatology-domain interpretation on the deduplicated ILPD cohort (`n=570`)."),
            ("code", setup_snippet + "from liver_disease.config import FIGURES_DIR, REPORTS_DIR\nfrom liver_disease.utils.io import load_json\n\neda = load_json(REPORTS_DIR / 'eda_summary.json')\nprint('Raw rows:', eda['raw_rows'], '| Deduplicated rows:', eda['deduplicated_rows'])\nprint('Gender chi-square test:', eda['gender_chi2_test'])\nprint('High collinearity pairs (|Spearman r| >= 0.70):')\ndisplay(pd.DataFrame(eda['high_collinearity_pairs_spearman_gt_0_7']))"),
            ("code", "tests_df = pd.DataFrame(eda['feature_statistical_tests_mann_whitney']).T\ndisplay(tests_df.sort_values('p_value'))"),
            ("code", "for img_name in [\n    '01_target_and_gender_distribution.png',\n    '02_numeric_distributions_by_class.png',\n    '03_boxplots_and_outliers.png',\n    '04_spearman_correlation_heatmap.png',\n]:\n    print(img_name)\n    display(Image(filename=str(FIGURES_DIR / img_name)))"),
        ],
        "03_preprocessing.ipynb": [
            ("md", "# 03 — Leakage-Free Splitting & Reproducible Preprocessing Pipeline\n\nVerifies deduplicated stratified 80/20 splitting, zero train/test feature overlap, `log1p` skewness handling, median imputation, scaling, binary gender encoding, and in-fold resampling safety."),
            ("code", setup_snippet + "from liver_disease.config import PROCESSED_DATA_DIR\nfrom liver_disease.data.loader import load_raw_dataset\nfrom liver_disease.data.preprocessing import (\n    build_preprocessor,\n    build_training_pipeline,\n    create_leakage_free_split,\n    get_transformed_feature_names,\n)\nfrom liver_disease.utils.io import load_json\n\ndf_raw = load_raw_dataset()\nX_train, X_test, y_train, y_test, split_meta = create_leakage_free_split(df_raw, save_artifacts=False)\nprint('Train shape:', X_train.shape, '| Positive rate:', round(float(y_train.mean()), 4))\nprint('Test shape: ', X_test.shape, '| Positive rate:', round(float(y_test.mean()), 4))\nprint('Train/Test exact feature overlap:', split_meta['train_test_feature_overlap'])"),
            ("code", "pre = build_preprocessor(scaler='robust', add_clinical_ratios=False, log_transform_skewed=True)\nX_train_t = pre.fit_transform(X_train)\nX_test_t = pre.transform(X_test)\nfeat_names = get_transformed_feature_names(pre)\nprint('Transformed feature names:', feat_names)\nprint('Train NaNs:', int(np.isnan(X_train_t).sum()), '| Test NaNs:', int(np.isnan(X_test_t).sum()))\ndisplay(pd.DataFrame(X_train_t, columns=feat_names).describe().T[['mean', 'std', 'min', '50%', 'max']].round(3))"),
        ],
        "04_model_benchmarking.ipynb": [
            ("md", "# 04 — Comprehensive 11-Model Benchmarking & Ablation Experiments\n\nAll experiments are conducted strictly on the 80% training split (`n=456`) using `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`. The holdout test set (`n=114`) is never touched."),
            ("code", setup_snippet + "from liver_disease.config import FIGURES_DIR, REPORTS_DIR\n\ndf_base = pd.read_csv(REPORTS_DIR / 'benchmark_11_models_baseline.csv')\nprint('=== Experiment 1: 11-Model Baseline Comparison ===')\ndisplay(df_base[['model', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean', 'f1_score_mean', 'recall_sensitivity_mean', 'specificity_mean', 'balanced_accuracy_mean', 'brier_score_mean']])\ndisplay(Image(filename=str(FIGURES_DIR / '05_model_benchmark_comparison.png')))"),
            ("code", "df_fe = pd.read_csv(REPORTS_DIR / 'benchmark_feature_engineering_ablation.csv')\nprint('=== Experiment 2: Feature Engineering Ablation ===')\ndisplay(df_fe.groupby('feature_set')[['roc_auc_mean', 'pr_auc_mean', 'balanced_accuracy_mean', 'brier_score_mean']].mean().round(4))\n\ndf_imb = pd.read_csv(REPORTS_DIR / 'benchmark_imbalance_strategies.csv')\nprint('=== Experiment 3: Class Imbalance Strategy Comparison (In-Fold Only) ===')\ndisplay(df_imb.groupby('imbalance_strategy')[['roc_auc_mean', 'pr_auc_mean', 'recall_sensitivity_mean', 'specificity_mean', 'balanced_accuracy_mean', 'f1_score_mean']].mean().round(4))\n\ndf_fs = pd.read_csv(REPORTS_DIR / 'benchmark_feature_selection.csv')\nprint('=== Experiment 4: Mutual-Information Feature Selection Ablation ===')\ndisplay(df_fs[['model', 'k_features', 'roc_auc_mean', 'pr_auc_mean', 'balanced_accuracy_mean']])"),
        ],
        "05_final_evaluation.ipynb": [
            ("md", "# 05 — Hyperparameter Tuning, Calibration, Single-Pass Holdout Evaluation & SHAP Explainability\n\nSummarizes Phase 5–6 validation tuning, calibration comparison, validation-only threshold selection, single-pass evaluation on the locked holdout test set (`n=114`), bootstrap 95% confidence intervals, and SHAP global + local explainability."),
            ("code", setup_snippet + "from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH, FIGURES_DIR, REPORTS_DIR\nfrom liver_disease.models.predict import LiverDiseasePredictor\nfrom liver_disease.utils.io import load_json\n\nreport = load_json(REPORTS_DIR / 'final_evaluation_report.json')\nmeta = report['model_metadata']\nev = meta['evaluation_metrics']\n\nprint('=== Tuned Candidates Repeated-CV Comparison ===')\ndf_tuned = pd.read_csv(REPORTS_DIR / 'tuned_candidates_cv_comparison.csv')\ndisplay(df_tuned[['candidate', 'selection_score', 'roc_auc_mean', 'pr_auc_mean', 'balanced_accuracy_mean', 'brier_score_mean']])\n\nprint('=== Calibration Comparison (CV) ===')\ndisplay(pd.DataFrame(ev['calibration_comparison'])[['calibration', 'brier_score_mean', 'roc_auc_mean', 'pr_auc_mean']])"),
            ("code", "print('Selected Model:', meta['model_name'], '| Calibration:', meta['calibration_method'], '| Selected Threshold:', meta['decision_threshold'])\nfor label, key in [\n    ('Holdout Test @ Default Threshold (0.50)', 'holdout_test_default_threshold_0_50'),\n    (f\"Holdout Test @ Validation-Selected Threshold ({meta['decision_threshold']:.2f})\", 'holdout_test_selected_threshold'),\n]:\n    m = ev[key]\n    print(f'\\n--- {label} ---')\n    for k in ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1_score', 'brier_score', 'confusion_matrix']:\n        print(f'  {k}: {m[k]}')\n\nprint('\\n=== Holdout Test 95% Bootstrap Confidence Intervals ===')\ndisplay(pd.DataFrame(ev['holdout_test_bootstrap_95_ci']).T)"),
            ("code", "for img_name in [\n    'final_test_roc_pr_curves.png',\n    'final_test_confusion_matrices.png',\n    'final_test_calibration_curve.png',\n    'shap_summary_beeswarm.png',\n    'shap_global_importance_bar.png',\n]:\n    print(img_name)\n    display(Image(filename=str(FIGURES_DIR / img_name)))"),
            ("code", "predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)\nsample_patient = {\n    'age': 62, 'gender': 'Male', 'total_bilirubin': 10.9, 'direct_bilirubin': 5.5,\n    'alkaline_phosphotase': 699.0, 'alamine_aminotransferase': 64.0,\n    'aspartate_aminotransferase': 100.0, 'total_proteins': 7.5,\n    'albumin': 3.2, 'albumin_and_globulin_ratio': 0.74,\n}\nexplained = predictor.explain_one(sample_patient, top_k=5)\nprint('Prediction:', explained['prediction'], explained['prediction_label'], '| Prob:', explained['probability'], '| Risk:', explained['risk_category'])\ndisplay(pd.DataFrame(explained['explanation']['top_contributing_features']))"),
        ],
    }

    written: List[Path] = []
    for filename, cell_specs in notebooks.items():
        path = NOTEBOOKS_DIR / filename
        nb = _make_nb(cell_specs)
        client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
        client.execute()
        with path.open("w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"Generated and executed: {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    build_notebooks()
