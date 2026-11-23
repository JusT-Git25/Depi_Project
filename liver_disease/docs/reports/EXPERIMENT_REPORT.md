# Experiment Report — Liver Disease Prediction System

## 1. Experimental Protocol & Leakage Prevention

All experiments were executed exclusively on the **Indian Liver Patient Dataset (ILPD)** (`583` raw rows, SHA-256 `dbb065e6bf2b226b6d9c04f066ddc0b45173861dfb3a4df83f3710d3927ef5de`) with strict anti-leakage discipline:

1. **Deduplication Before Splitting**: The data audit (`artifacts/reports/data_audit_report.json`) uncovered **13 exact duplicate rows**, all occurring as consecutive adjacent pairs with **0 target conflicts**. Keeping exact duplicates would allow identical patient records to fall into both training and test partitions. Deduplicating via `drop_duplicates(keep="first")` yielded **570 unique observations** (`406` Liver Disease = `71.23%`, `164` No Liver Disease = `28.77%`).
2. **Locked Holdout Test Split**: Stratified 80/20 split (`random_state=42`):
   - **Training / Validation Partition (`data/processed/train.csv`)**: `456` records (`325` positive = `71.27%`, `131` negative = `28.73%`).
   - **Holdout Test Partition (`data/processed/test.csv`)**: `114` records (`81` positive = `71.05%`, `33` negative = `28.95%`).
   - **Verified Train/Test Exact Feature Overlap**: `0`.
3. **In-Fold Preprocessing & Resampling**: Median imputation (`A/G Ratio` has 4 missing values), `log1p` scaling of right-skewed biomarkers, `RobustScaler`, binary one-hot encoding (`Gender`), and resampling (`RandomOverSampler`, `SMOTE`) were fitted strictly inside each training fold.
4. **Cross-Validation Strategy**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` (15 validation folds per configuration) to reduce variance on `n=456` tabular data.
5. **Pre-Registered Selection Criteria**:
   - Primary ranking: Composite discrimination score `0.5 × ROC-AUC + 0.5 × PR-AUC` with Brier score tie-breaker (avoiding accuracy traps under 71/29 imbalance).
   - Threshold optimization: Selected exclusively on out-of-fold validation probabilities maximizing screening utility `0.5 × Recall + 0.5 × F1` subject to a minimum specificity guardrail (`Specificity >= 0.50`).
   - Holdout test set touched **exactly once** after model, features, hyperparameters, calibration, and threshold were frozen.

---

## 2. Experiment 1 — 11-Algorithm Baseline Benchmark (5×3 Repeated Stratified CV)

Source: `artifacts/reports/benchmark_11_models_baseline.csv` | Visualization: `artifacts/reports/figures/05_model_benchmark_comparison.png`

| Rank | Model | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | F1-Score | Recall (Sens.) | Specificity | Balanced Acc. | Accuracy | Brier |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ExtraTrees** | **0.7343 ± 0.0477** | **0.8876 ± 0.0201** | 0.8186 | 0.9303 | 0.1526 | 0.5414 | 0.7069 | **0.1781** |
| 2 | **KNN** (`k=11`, distance) | 0.7279 ± 0.0455 | 0.8780 ± 0.0295 | 0.8008 | 0.8574 | 0.3000 | 0.5787 | 0.6973 | 0.1818 |
| 3 | **LogisticRegression** | 0.7228 ± 0.0485 | 0.8791 ± 0.0261 | 0.8143 | 0.9036 | 0.2186 | 0.5611 | 0.7069 | 0.1800 |
| 4 | **RandomForest** | 0.7211 ± 0.0454 | 0.8795 ± 0.0244 | 0.8065 | 0.8779 | 0.2595 | 0.5687 | 0.7003 | 0.1816 |
| 5 | **CatBoost** | 0.7074 ± 0.0486 | 0.8710 ± 0.0256 | 0.7945 | 0.8462 | 0.2972 | 0.5717 | 0.6885 | 0.1963 |
| 6 | **GradientBoosting** | 0.7046 ± 0.0569 | 0.8673 ± 0.0311 | 0.7998 | 0.8482 | 0.3255 | **0.5868** | 0.6981 | 0.1961 |
| 7 | **XGBoost** | 0.7015 ± 0.0539 | 0.8685 ± 0.0266 | 0.7854 | 0.8226 | 0.3251 | 0.5738 | 0.6798 | 0.2043 |
| 8 | **HistGradientBoosting** | 0.6991 ± 0.0528 | 0.8688 ± 0.0294 | 0.7810 | 0.8236 | 0.2951 | 0.5593 | 0.6718 | 0.2008 |
| 9 | **LightGBM** | 0.6988 ± 0.0466 | 0.8697 ± 0.0197 | 0.7888 | 0.8246 | **0.3406** | 0.5826 | 0.6856 | 0.2258 |
| 10 | **SVM (RBF)** | 0.6348 ± 0.0501 | 0.8119 ± 0.0296 | 0.8322 | 1.0000 | **0.0000** | 0.5000 | 0.7127 | 0.2013 |
| 11 | **DecisionTree** | 0.6187 ± 0.0539 | 0.7973 ± 0.0344 | 0.7545 | 0.7908 | 0.2546 | 0.5227 | 0.6368 | 0.2564 |

### Key Insights from Experiment 1
- **Why Accuracy Alone Fails**: Unweighted `SVM_RBF` achieved the highest raw accuracy (`71.27%`) and F1 (`0.8322`) strictly by predicting **100% of patients as diseased** (`Specificity = 0.000`, `Balanced Accuracy = 0.500`), while having the second-worst ROC-AUC (`0.6348`).
- **Small-Sample Bias-Variance Tradeoff**: On `n=456` noisy clinical records, randomized bagging (`ExtraTrees`, `RandomForest`) and regularized linear models (`LogisticRegression`) outperformed higher-variance boosting algorithms (`XGBoost`, `LightGBM`, `HistGradientBoosting`) in both discrimination and Brier score.
- **Default `0.50` Threshold Behavior**: Because Liver Disease is the `71.3%` majority class, unweighted models at `0.50` exhibit high sensitivity (`82–93%`) but low specificity (`15–34%`), motivating imbalance weighting and validation-based threshold selection.

---

## 3. Experiment 2 — Feature Engineering Ablation

Compared 3 representations across 5 diverse model families (`LogisticRegression`, `RandomForest`, `ExtraTrees`, `CatBoost`, `SVM_RBF`). Source: `artifacts/reports/benchmark_feature_engineering_ablation.csv`.

| Feature Configuration | Mean ROC-AUC | Mean PR-AUC | Decision |
| :--- | :---: | :---: | :--- |
| `raw_only` (10 original features) | 0.7011 | 0.8678 | Baseline |
| **`log1p_only`** (`log1p` on `TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`) | **0.7041** | **0.8658** | **Selected** (highest aggregate score, stabilizes skew without adding collinearity) |
| `log1p_plus_clinical_ratios` (+ AST/ALT, DB/TB, Indirect Bilirubin, Globulin) | 0.7048 | 0.8636 | Not adopted (no consistent gain over `log1p_only`, increases collinearity) |

---

## 4. Experiment 3 — Class Imbalance Strategy Comparison (In-Fold Only)

Compared `none`, `class_weight`, `RandomOverSampler` (in-fold), and `SMOTE` (in-fold) across 6 algorithms (`LogisticRegression`, `RandomForest`, `ExtraTrees`, `LightGBM`, `CatBoost`, `XGBoost`). Source: `artifacts/reports/benchmark_imbalance_strategies.csv`.

| Imbalance Strategy | Mean ROC-AUC | Mean PR-AUC | Mean Recall | Mean Specificity | Mean Balanced Acc. | Mean F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`class_weight`** | **0.7154** | **0.8762** | 0.6990 | **0.5976** | **0.6483** | 0.7488 |
| `none` | 0.7143 | 0.8759 | 0.8675 | 0.2656 | 0.5666 | 0.8013 |
| `smote` (in-fold) | 0.7144 | 0.8731 | 0.7130 | 0.5548 | 0.6339 | 0.7511 |
| `random_oversampler` (in-fold) | 0.7124 | 0.8750 | 0.7192 | 0.5517 | 0.6354 | 0.7550 |

**Finding**: Native `class_weight="balanced"` achieved the highest mean ROC-AUC, PR-AUC, Specificity, and Balanced Accuracy without synthetic interpolation artifacts. In Phase 5, `class_weight` (`None` vs `"balanced"`) was included inside each tuned candidate's hyperparameter grid so CV selected the optimal weighting per model family.

---

## 5. Experiment 4 — Feature Selection Ablation

Tested in-fold `SelectKBest(mutual_info_classif)` for `k ∈ {6, 8, 10, all}` across `LogisticRegression`, `RandomForest`, and `CatBoost`. Source: `artifacts/reports/benchmark_feature_selection.csv`.
- `k=8` modestly improved CV ROC-AUC for `LogisticRegression` (`0.7259` vs `0.7228`), `RandomForest` (`0.7294` vs `0.7211`), and `CatBoost` (`0.7230` vs `0.7074`), while `k=6` degraded performance.
- Because all 10 features are standard routine LFT panel measurements collected simultaneously in clinical practice and `SoftVoting_ET_RF_LR` on all 10 features already achieved `0.7417` ROC-AUC / `0.8912` PR-AUC, all 10 predictors were retained for full clinical transparency and SHAP attribution coverage.

---

## 6. Experiment 5 — Hyperparameter Tuning & Soft-Voting Ensemble (5×3 Repeated CV)

Source: `artifacts/reports/tuned_candidates_cv_comparison.csv`

| Rank | Candidate | Selection Score | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Brier Score | Key Tuned Hyperparameters |
| ---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **`SoftVoting_ET_RF_LR`** | **0.8165** | **0.7417 ± 0.0469** | **0.8912 ± 0.0213** | **0.1765** | Soft vote `[2×ET, 1×RF, 1×LR]` |
| 2 | `ExtraTrees_Tuned` | 0.8158 | 0.7408 ± 0.0502 | 0.8908 ± 0.0198 | 0.1797 | `n_estimators=300, max_depth=14, min_samples_leaf=1, max_features='sqrt', class_weight='balanced'` |
| 3 | `LogisticRegression_Tuned` | 0.8079 | 0.7328 ± 0.0473 | 0.8830 ± 0.0263 | 0.1774 | `C=0.05, class_weight=None` |
| 4 | `CatBoost_Tuned` | 0.8048 | 0.7289 ± 0.0468 | 0.8807 ± 0.0256 | 0.2007 | `iterations=200, depth=5, lr=0.02, l2_leaf_reg=3.0, auto_class_weights='Balanced'` |
| 5 | `RandomForest_Tuned` | 0.8018 | 0.7228 ± 0.0423 | 0.8809 ± 0.0228 | 0.1820 | `n_estimators=500, max_depth=None, min_samples_leaf=2, max_features='sqrt', class_weight=None` |

**Ensemble Justification**: Combining tuned `ExtraTrees` (high-randomness balanced trees), `RandomForest` (bagged trees), and `LogisticRegression` (L2-regularized smooth linear probabilities) improved **every primary CV criterion simultaneously**: highest ROC-AUC (`0.7417`), lowest ROC-AUC variance vs ExtraTrees (`±0.0469` vs `±0.0502`), highest PR-AUC (`0.8912`), and lowest Brier score (`0.1765`).

---

## 7. Experiment 6 — Probability Calibration & Validation Threshold Selection

### Calibration Study (5×3 Repeated CV on `SoftVoting_ET_RF_LR`)
| Calibration Method | Brier Score (Mean ± SD) | ROC-AUC | PR-AUC | Decision |
| :--- | :---: | :---: | :---: | :--- |
| **`none` (Unwrapped Soft-Voting)** | **0.1765 ± 0.0116** | 0.7417 | 0.8912 | **Selected** (lowest Brier score) |
| `sigmoid` (Platt Scaling, 5-fold) | 0.1783 ± 0.0093 | 0.7418 | 0.8913 | Slightly worse Brier (`+0.0018`) |
| `isotonic` (5-fold) | 0.1791 ± 0.0129 | 0.7363 | 0.8853 | Overfits small sample (`n=456`) |

### Out-of-Fold Validation Threshold Optimization (`X_train` Only)
- Default threshold `0.50`: OOF Recall = `0.8738`, Specificity = `0.2977`, Balanced Accuracy = `0.5858`, F1 = `0.8103`.
- **Selected threshold `0.57`** (`max_screening_utility_with_specificity_ge_0.50`): OOF Recall = `0.7600`, Specificity = `0.5038`, Precision = `0.7917`, Balanced Accuracy = `0.6319`, F1 = `0.7755`.

---

## 8. Final Single-Pass Holdout Test Evaluation (`n = 114`)

Source: `artifacts/reports/final_evaluation_report.json`

| Metric | Default Threshold (`0.50`) | Validation-Selected Threshold (`0.57`) | 95% Bootstrap CI (`@ 0.57`, 1000 resamples) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8066 | **0.8066** | `0.7233 – 0.8798` |
| **PR-AUC** | 0.9233 | **0.9233** | `0.8769 – 0.9582` |
| **Brier Score** | 0.1571 | **0.1571** | — |
| **Accuracy** | 0.7368 | **0.7281** | `0.6404 – 0.8070` |
| **Balanced Accuracy** | 0.5993 | **0.6650** | `0.5654 – 0.7567` |
| **Precision (PPV)** | 0.7576 | **0.8049** | `0.7142 – 0.8824` |
| **Recall / Sensitivity** | 0.9259 | **0.8148** | `0.7284 – 0.9024` |
| **Specificity (TNR)** | 0.2727 | **0.5152** | `0.3427 – 0.6897` |
| **F1-Score** | 0.8333 | **0.8098** | `0.7403 – 0.8715` |
| **F2-Score** | 0.8865 | **0.8128** | — |
| **Confusion Matrix (`TN, FP, FN, TP`)** | `9, 24, 6, 75` | **`17, 16, 15, 66`** | — |

### False Negative vs. False Positive Analysis
- At threshold `0.57`, the ensemble captures **81.5% (`66/81`)** of liver disease cases while nearly doubling specificity (`51.5%` vs `27.3%` at `0.50`) and raising precision to `80.5%`.
- In an even higher-sensitivity screening research scenario where minimizing False Negatives (`FN=6` vs `FN=15`) outweighs false-alarm follow-up testing (`FP=24` vs `FP=16`), the documented default threshold `0.50` (`Recall = 92.6%`) illustrates the exact operating trade-off. Because `probability` is always returned alongside `prediction`, downstream research workflows can inspect the continuous risk score directly.

---

## 9. Global SHAP Explainability Ranking (`KernelExplainer`)

| Rank | Feature | Mean `|SHAP|` | Clinical Directionality |
| ---: | :--- | :---: | :--- |
| 1 | `DB` (Direct Bilirubin) | `0.05043` | Higher conjugated bilirubin strongly increases risk score |
| 2 | `TB` (Total Bilirubin) | `0.03957` | Elevated total bilirubin strongly increases risk score |
| 3 | `Sgpt` (ALT) | `0.03688` | Elevated transaminase increases risk score |
| 4 | `Alkphos` (ALP) | `0.03562` | Elevated alkaline phosphatase increases risk score |
| 5 | `Sgot` (AST) | `0.03044` | Elevated AST increases risk score |
| 6 | `Age` | `0.02556` | Older age modestly elevates predicted risk score |
| 7 | `TP` (Total Proteins) | `0.01589` | Secondary context marker |
| 8 | `A/G Ratio` | `0.01369` | Lower A/G ratio elevates risk score |
| 9 | `ALB` (Albumin) | `0.01219` | Lower albumin elevates risk score |
| 10 | `Gender_Male` | `0.00742` | Minor demographic effect |

> **Attribution Limitation**: Strongly correlated markers (`TB`/`DB` $\rho=0.958$, `Sgpt`/`Sgot` $\rho=0.772$, `TP`/`ALB` $\rho=0.778$) share SHAP credit. SHAP values describe model behavior on ILPD and are **not** evidence of biological causation.
