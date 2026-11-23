# Data Dictionary — Indian Liver Patient Dataset (ILPD)

## 1. Dataset Provenance

| Attribute | Verified Value |
| :--- | :--- |
| **Dataset Name** | Indian Liver Patient Dataset (ILPD) |
| **Source** | UCI Machine Learning Repository (Dataset ID 225) |
| **Source URL** | https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset |
| **Original Authors** | Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu (2012) |
| **Geographic Origin** | North East of Andhra Pradesh, India |
| **Canonical File** | `data/raw/Indian Liver Patient Dataset (ILPD).csv` |
| **SHA-256 Checksum** | `dbb065e6bf2b226b6d9c04f066ddc0b45173861dfb3a4df83f3710d3927ef5de` |
| **File Size** | 23,817 bytes |
| **Raw Dimensions** | 583 records × 11 columns (10 predictors + 1 binary target) |
| **Deduplicated Dimensions** | 570 unique records (13 consecutive exact duplicate rows removed, 0 target conflicts) |

---

## 2. Feature Definitions, Medical Units, and Observed Distributions

All summary statistics below are verified directly against the raw 583-record ILPD CSV (`artifacts/reports/data_audit_report.json`). Mann-Whitney U $p$-values and Spearman rank correlations ($\rho$) are computed on the deduplicated 570-record cohort (`artifacts/reports/eda_summary.json`).

| Raw CSV Column | Canonical API Field | Data Type | Medical Unit | Normal Clinical Reference | Observed Range (ILPD) | Median (IQR) / Counts | Missing | API Allowed Validation Bounds | Spearman $\rho$ with Disease ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Age` | `age` | `int` | years | 1–120 years | 4 – 90 *(>89 capped at 90)* | 45.0 (33.0–58.0) | 0 (0.00%) | 1 – 120 | `+0.130` (`2.02e-03`) |
| `Gender` | `gender` | `category` | categorical | `Male` / `Female` | Male: 441 (75.6%), Female: 142 (24.4%) | Male / Female | 0 (0.00%) | `"Male"` or `"Female"` | $\chi^2 = 3.12$ (`p=0.0773`) |
| `TB` | `total_bilirubin` | `float` | mg/dL | 0.1 – 1.2 mg/dL | 0.4 – 75.0 | 1.0 (0.8–2.6) | 0 (0.00%) | 0.05 – 100.0 | `+0.316` (`4.68e-14`) |
| `DB` | `direct_bilirubin` | `float` | mg/dL | 0.0 – 0.3 mg/dL | 0.1 – 19.7 | 0.3 (0.2–1.3) | 0 (0.00%) | 0.0 – 50.0 | `+0.309` (`1.58e-13`) |
| `Alkphos` | `alkaline_phosphotase` | `float` | IU/L | 44 – 147 IU/L | 63 – 2110 | 208.0 (175.5–298.0) | 0 (0.00%) | 10.0 – 5000.0 | `+0.274` (`6.23e-11`) |
| `Sgpt` | `alamine_aminotransferase` | `float` | IU/L | 7 – 56 IU/L | 10 – 2000 | 35.0 (23.0–60.5) | 0 (0.00%) | 1.0 – 5000.0 | `+0.291` (`4.11e-12`) |
| `Sgot` | `aspartate_aminotransferase` | `float` | IU/L | 10 – 40 IU/L | 10 – 4929 | 42.0 (25.0–87.0) | 0 (0.00%) | 1.0 – 10000.0 | `+0.310` (`1.30e-13`) |
| `TP` | `total_proteins` | `float` | g/dL | 6.0 – 8.3 g/dL | 2.7 – 9.6 | 6.6 (5.8–7.2) | 0 (0.00%) | 1.0 – 20.0 | `-0.035` (`3.98e-01`) |
| `ALB` | `albumin` | `float` | g/dL | 3.5 – 5.0 g/dL | 0.9 – 5.5 | 3.1 (2.6–3.8) | 0 (0.00%) | 0.5 – 12.0 | `-0.174` (`3.47e-05`) |
| `A/G Ratio` | `albumin_and_globulin_ratio` | `float` | ratio (dimensionless) | 1.0 – 2.5 | 0.3 – 2.8 | 0.93 (0.7–1.1) | **4 (0.69%)** | 0.1 – 10.0 | `-0.197` (`2.82e-06`) |

---

## 3. Target Variable Definition

| Column | Raw Value | Meaning | Internal Binary Class | Raw Count (%) | Deduplicated Count (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Selector` | `1` | **Liver Disease** | `1` (Positive) | 416 (71.36%) | 406 (71.23%) |
| `Selector` | `2` | **No Liver Disease** | `0` (Negative) | 167 (28.64%) | 164 (28.77%) |

- **Class Imbalance Ratio**: `2.49 : 1` (raw) / `2.48 : 1` (deduplicated).
- **API Output Encoding**: Both `prediction` (`1` or `2`, matching original ILPD encoding) and `prediction_label` (`"Liver Disease"` or `"No Liver Disease"`) are returned explicitly.

---

## 4. Hepatology Domain Notes & Inter-Feature Collinearity

1. **Bilirubin Axis (`TB`, `DB`)**: Strongly collinear ($\rho = +0.958$). Total bilirubin combines conjugated (direct) and unconjugated (indirect) fractions. Elevated bilirubin reflects impaired hepatic clearance or biliary obstruction.
2. **Hepatocellular Injury Enzymes (`Sgpt`/ALT, `Sgot`/AST)**: Strongly correlated ($\rho = +0.772$) and heavily right-skewed (`skewness = 6.55` and `10.55`). Sharp elevations reflect hepatocyte necrosis/inflammation.
3. **Cholestatic Enzyme (`Alkphos`/ALP)**: Right-skewed (`skewness = 3.77`); elevated in biliary tract obstruction and infiltrative liver disorders.
4. **Hepatic Synthetic Function (`TP`, `ALB`, `A/G Ratio`)**: `TP` and `ALB` correlate at $\rho = +0.778$; `ALB` and `A/G Ratio` correlate at $\rho = +0.754$. Lower serum albumin and lower A/G ratio are negatively associated with liver disease ($\rho = -0.174$ and $-0.197$), reflecting reduced hepatic protein synthesis in chronic liver impairment.

---

## 5. Documented Data Quality Notes & Cleaning Decisions

- **Missing Values (4 cells)**: Located strictly in `A/G Ratio` at 0-based raw indices `[209, 241, 253, 312]`. Handled by `SimpleImputer(strategy="median")` fitted strictly inside training folds (never across the full dataset). At inference time, the REST API strictly requires all 10 features and rejects missing values with HTTP 422.
- **Exact Duplicates (13 rows)**: 13 consecutive adjacent duplicate pairs (`[18,19]`, `[25,26]`, `[33,34]`, `[54,55]`, `[61,62]`, `[105,106]`, `[107,108]`, `[137,138]`, `[142,143]`, `[157,158]`, `[163,164]`, `[173,174]`, `[200,201]`), all with 0 target conflicts. Removed via `drop_duplicates(keep="first")` prior to train/test splitting to guarantee zero identical-record leakage.
- **Preserved Anomalies**:
  - 3 rows (`[246, 261, 279]`) record `DB > TB`.
  - 8 rows have reported `A/G Ratio` differing from `ALB / (TP - ALB)` by `> 0.25`.
  - 11–14% of bilirubin/enzyme measurements exceed `1.5×IQR` fences due to acute clinical presentations.
  - None of these raw medical values were altered or deleted; `log1p` transformation compresses extreme positive skew while preserving rank ordering and clinical signal.
