# ILPD Dataset Documentation & Provenance

## Source & Provenance

- **Dataset Name**: Indian Liver Patient Dataset (ILPD)
- **Canonical File**: `data/raw/Indian Liver Patient Dataset (ILPD).csv`
- **SHA-256 Checksum**: `dbb065e6bf2b226b6d9c04f066ddc0b45173861dfb3a4df83f3710d3927ef5de`
- **File Size**: 23,817 bytes
- **Original Source**: UCI Machine Learning Repository (Dataset ID 225) — https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- **Original Creators**: Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu (2012)
- **Cohort Origin**: Collected from North East of Andhra Pradesh, India.
- **Compliance**: Only the original ILPD dataset is used. No external datasets were merged or added.

## Verified Structure

- **Raw Shape**: 583 records × 11 columns (10 predictors + 1 binary target `Selector`)
- **Target Encoding**:
  - `1` = Liver Disease (416 records, 71.36%) → internal positive class `y = 1`
  - `2` = No Liver Disease (167 records, 28.64%) → internal negative class `y = 0`
  - Imbalance Ratio (Positive : Negative): `2.49 : 1`

## Verified Data Quality Findings (`artifacts/reports/data_audit_report.json`)

1. **Missing Values**: Exactly **4 missing cells** (0.69%), all in `A/G Ratio` at 0-based raw row indices `[209, 241, 253, 312]`. Zero missing values in all other columns.
   - *Decision*: Median imputation fitted strictly inside training folds (no global imputation prior to splitting).
2. **Exact Duplicates**: **13 exact duplicate extra rows** (13 pairs, all occurring as consecutive adjacent rows in the raw file, with **0 target conflicts**).
   - *Decision*: Deduplicate exact rows (`keep="first"`) prior to train/test splitting, leaving **570 unique observations** (406 disease, 164 non-disease) and guaranteeing zero identical-record leakage between training, validation, and test partitions.
3. **Near-Duplicates**: 12 pairs had standardized Euclidean distance `< 0.25`, of which 6 pairs had opposite target labels (`Selector=1` vs `Selector=2`). These represent distinct patients with similar routine lab panels and are preserved intact.
4. **Domain Consistency Anomalies (Preserved Without Silent Modification)**:
   - 3 raw rows (`[246, 261, 279]`) record `DB > TB` (minor lab measurement/entry discrepancy).
   - 8 raw rows have reported `A/G Ratio` differing from `ALB / (TP - ALB)` by `> 0.25`.
   - *Decision*: Per strict medical data integrity rules, raw observations are **not** silently edited or deleted.
5. **Heavy Right Skewness & Extreme Values**:
   - `TB` (skew 4.91), `DB` (3.21), `Alkphos` (3.77), `Sgpt` (6.55), and `Sgot` (10.55) exhibit 11–14% values above the standard `1.5×IQR` upper fence.
   - *Decision*: Extreme elevations of bilirubin and hepatic transaminases are authentic markers of acute/severe liver injury and are **never** discarded; `log1p` transformation and `RobustScaler` are evaluated inside CV pipelines instead.

## Processed Splits (`data/processed/`)

- `train.csv`: 80% stratified training partition (456 unique rows) used for all cross-validation, hyperparameter tuning, calibration, and threshold selection.
- `test.csv`: 20% stratified holdout test partition (114 unique rows) locked until final single-pass evaluation.
- `split_metadata.json`: Exact deduplication indices, split sizes, class balances, and verified zero train/test overlap check.
