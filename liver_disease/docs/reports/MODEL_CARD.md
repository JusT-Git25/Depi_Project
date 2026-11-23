# Model Card — Liver Disease Prediction Pipeline (`v1.0.0`)

## 1. Model Overview

| Attribute | Value |
| :--- | :--- |
| **Model Name** | `SoftVoting_ET_RF_LR` |
| **Model Type** | `sklearn.ensemble.VotingClassifier` (soft voting, weights `[2, 1, 1]`) |
| **Ensemble Members** | 1) Tuned `ExtraTreesClassifier` (`class_weight="balanced"`) · 2) Tuned `RandomForestClassifier` · 3) Tuned `LogisticRegression` (`C=0.05`) |
| **Model Version** | `1.0.0` |
| **Artifact File** | `artifacts/models/liver_disease_pipeline.joblib` (5,749,315 bytes) |
| **Trained Timestamp (UTC)** | `2026-11-23T00:00:55.548716+00:00` |
| **Calibration Method** | `none` (unwrapped soft-voting achieved lowest CV Brier score `0.1765` vs `sigmoid` `0.1783` and `isotonic` `0.1791`) |
| **Default Threshold** | `0.50` |
| **Production Decision Threshold** | `0.57` (selected strictly on out-of-fold validation predictions with `Specificity >= 0.50` guardrail) |
| **Risk Category Bands** | `Low`: `< 0.35` · `Moderate`: `0.35 – < 0.65` · `High`: `>= 0.65` |
| **Explainability** | SHAP `KernelExplainer` with embedded `kmeans(30)` background from ILPD training data |

---

## 2. Intended Use & Out-of-Scope Warning

### Intended Use
- Academic graduation project demonstrating production-grade Machine Learning Engineering, leakage-free validation, SHAP explainability, secure FastAPI microservice design, automated QA testing, and Dockerized deployment.
- Research prototyping and backend integration demonstration using liver function test (LFT) tabular attributes.

### Explicitly Out of Scope (CRITICAL)
- **NOT a clinically validated diagnostic tool or medical device.**
- **MUST NOT** be used by clinicians, patients, or health systems to diagnose liver disease, prescribe treatment, triage emergencies, or rule out pathology.
- **Probabilities are NOT clinically calibrated diagnostic probabilities.** They represent statistical risk scores learned from a single small regional dataset (`n=570` deduplicated records).

---

## 3. Training & Evaluation Data

- **Dataset**: UCI Indian Liver Patient Dataset (ILPD) (`data/raw/Indian Liver Patient Dataset (ILPD).csv`, SHA-256 `dbb065e6bf2b226b6d9c04f066ddc0b45173861dfb3a4df83f3710d3927ef5de`).
- **Cohort**: 583 raw records collected in North East Andhra Pradesh, India; 13 consecutive exact duplicate records removed (`0` target conflicts) → **570 unique records**.
- **Train / Holdout Split**: Stratified 80/20 split (`random_state=42`): `456` training rows (`71.27%` positive) and `114` locked holdout test rows (`71.05%` positive), with `0` feature-row overlap.
- **Preprocessing**: Row-wise `log1p` on right-skewed serum biomarkers (`TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`), training-fold median imputation for missing numerical values, `RobustScaler`, and binary one-hot encoding for `Gender`.

---

## 4. Quantitative Performance Summary

### Cross-Validation (`5×3` Repeated Stratified CV, `n = 456`)
- **ROC-AUC**: `0.7417 ± 0.0469`
- **PR-AUC**: `0.8912 ± 0.0213`
- **Brier Score**: `0.1765 ± 0.0116`

### Locked Holdout Test Set (`n = 114`, Single-Pass Evaluation)
| Metric | @ Selected Threshold (`0.57`) | 95% Bootstrap CI | @ Default Threshold (`0.50`) |
| :--- | :---: | :---: | :---: |
| ROC-AUC | **0.8066** | `0.7233 – 0.8798` | 0.8066 |
| PR-AUC | **0.9233** | `0.8769 – 0.9582` | 0.9233 |
| Brier Score | **0.1571** | — | 0.1571 |
| Accuracy | **0.7281** | `0.6404 – 0.8070` | 0.7368 |
| Balanced Accuracy | **0.6650** | `0.5654 – 0.7567` | 0.5993 |
| Precision | **0.8049** | `0.7142 – 0.8824` | 0.7576 |
| Recall (Sensitivity) | **0.8148** | `0.7284 – 0.9024` | 0.9259 |
| Specificity | **0.5152** | `0.3427 – 0.6897` | 0.2727 |
| F1-Score | **0.8098** | `0.7403 – 0.8715` | 0.8333 |
| Confusion Matrix (`TN, FP, FN, TP`) | `17, 16, 15, 66` | — | `9, 24, 6, 75` |

---

## 5. Known Limitations, Bias & Ethical Considerations

1. **Small Sample Size & Wide Confidence Intervals**: With only `570` deduplicated records (`114` in the holdout test set), performance estimates carry noticeable statistical uncertainty (e.g., Specificity 95% CI spans `34.3%–69.0%`). Note that holdout ROC-AUC (`0.8066`) exceeded CV ROC-AUC (`0.7417`) due to natural small-sample variance; the CV estimate (`~0.74`) is the more conservative generalization expectation.
2. **Demographic & Geographic Skew**: The ILPD cohort is `75.6%` male (`441/583`) and drawn exclusively from one region in Andhra Pradesh, India (2012). Performance may not generalize to other demographics, age groups, laboratory assays, or clinical prevalences.
3. **Class Imbalance & High Prevalence Prior**: ILPD has `71.3%` liver disease prevalence (typical of a hepatology clinic referral cohort, NOT the general population). Applying this model to a low-prevalence general population would yield substantially higher false-positive rates and miscalibrated probabilities.
4. **Residual False Negatives (`15/81` at `0.57`)**: Roughly `18.5%` of liver disease records in the holdout test set scored below `0.57` because early/mild liver disease can present with near-normal routine LFT panels.
5. **No Clinical Outcome Subtypes**: ILPD collapses all liver pathologies (viral hepatitis, NAFLD, alcoholic liver disease, cirrhosis, cholestasis) into a single binary label (`1` vs `2`) and lacks key clinical context (alcohol history, viral serology, BMI, imaging, medications).
6. **SHAP Correlation Caveat**: Strong collinearity among bilirubin (`TB`/`DB`) and transaminases (`Sgpt`/`Sgot`) distributes attribution across correlated features; SHAP values must never be interpreted as causal medical mechanisms.
