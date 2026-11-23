# Liver Disease Prediction System — User Input & Integration Guide

> **Direct Summary:** To get a prediction, the user must enter **exactly 10 fields**: `Age` (`age`), `Gender` (`gender`), `Total Bilirubin` (`total_bilirubin`), `Direct Bilirubin` (`direct_bilirubin`), `Alkaline Phosphotase` (`alkaline_phosphotase`), `Alamine Aminotransferase` (`alamine_aminotransferase`), `Aspartate Aminotransferase` (`aspartate_aminotransferase`), `Total Proteins` (`total_proteins`), `Albumin` (`albumin`), and `A/G Ratio` (`albumin_and_globulin_ratio`). All 10 fields are strictly required.
>
> **Academic Research Notice:** As stated in the application (`frontend/app.py` and `src/liver_disease/config.py`), this system is an academic research prototype trained on the UCI Indian Liver Patient Dataset (ILPD) and is **not** a clinically validated medical diagnostic tool.

---

## Verification Scope (Inspected Components)

Every fact, field name, unit, validation bound, and transformation in this guide was verified directly from the project's actual implementation and serialized model artifact:

| Layer | Inspected File(s) |
| :--- | :--- |
| **Streamlit Frontend** | `frontend/app.py` |
| **Central Configuration & Feature Specs** | `src/liver_disease/config.py` |
| **FastAPI Endpoints & Authentication** | `src/liver_disease/api/main.py`, `src/liver_disease/api/routes.py`, `src/liver_disease/api/security.py` |
| **Pydantic Request/Response Schemas** | `src/liver_disease/api/schemas.py` |
| **Data Loading & Validation** | `src/liver_disease/data/loader.py`, `src/liver_disease/data/validation.py` |
| **Preprocessing & Feature Engineering** | `src/liver_disease/data/preprocessing.py`, `src/liver_disease/features/engineering.py` |
| **Model Training & Inference Pipeline** | `src/liver_disease/models/train.py`, `src/liver_disease/models/predict.py`, `src/liver_disease/models/explain.py` |
| **Trained Model Artifact & Metadata** | `artifacts/models/liver_disease_pipeline.joblib`, `artifacts/models/model_metadata.json`, `artifacts/preprocessing/preprocessing_schema.json` |

---

## 1. Exact User Input Fields

The table below lists **all 10 patient input fields** required to obtain a liver disease prediction.

| # | Field Displayed in Application | Feature Name (API / Raw ILPD) | What It Means | Data Type | Expected Unit | Required/Optional | Validation (Enforced in Code) |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | `Age (years)` | `age` / `Age` | Patient's age in whole years (note: patients >89 were recorded as 90 in UCI ILPD). | Integer (`int`) — Streamlit `number_input` (`step=1`, default `45`) | `years` | **Required** | Min `1`, Max `120` (inclusive). Rejects booleans and fractional decimals (e.g., `45.5`); whole-number floats (e.g., `45.0`) are coerced to `int`. |
| **2** | `Gender` | `gender` / `Gender` | Patient's biological sex / gender recorded for the model. | Categorical string (`Literal["Male", "Female"]`) — Streamlit `selectbox` (default `"Male"`) | Categorical (`Male` / `Female`) | **Required** | Must be exactly `"Male"` or `"Female"` (case-sensitive). Leading/trailing whitespace is stripped in API before checking. |
| **3** | `Total Bilirubin — TB (mg/dL) [Ref: 0.1–1.2]` | `total_bilirubin` / `TB` | Total serum bilirubin concentration (conjugated + unconjugated). | Float (`float`) — Streamlit `number_input` (`step=0.1`, default `1.20`) | `mg/dL` | **Required** | Min `0.05`, Max `100.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **4** | `Direct Bilirubin — DB (mg/dL) [Ref: 0.0–0.3]` | `direct_bilirubin` / `DB` | Conjugated (direct) serum bilirubin concentration. | Float (`float`) — Streamlit `number_input` (`step=0.1`, default `0.40`) | `mg/dL` | **Required** | Min `0.0`, Max `50.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **5** | `Alkaline Phosphotase — ALP (IU/L) [Ref: 44–147]` | `alkaline_phosphotase` / `Alkphos` | Serum alkaline phosphatase enzyme level (cholestatic marker). | Float (`float`) — Streamlit `number_input` (`step=1.0`, default `210.0`) | `IU/L` | **Required** | Min `10.0`, Max `5000.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **6** | `Alamine Aminotransferase — ALT/SGPT (IU/L) [Ref: 7–56]` | `alamine_aminotransferase` / `Sgpt` | Serum alanine aminotransferase enzyme level (hepatocellular injury marker). | Float (`float`) — Streamlit `number_input` (`step=1.0`, default `38.0`) | `IU/L` | **Required** | Min `1.0`, Max `5000.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **7** | `Aspartate Aminotransferase — AST/SGOT (IU/L) [Ref: 10–40]` | `aspartate_aminotransferase` / `Sgot` | Serum aspartate aminotransferase enzyme level (hepatocellular injury marker). | Float (`float`) — Streamlit `number_input` (`step=1.0`, default `45.0`) | `IU/L` | **Required** | Min `1.0`, Max `10000.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **8** | `Total Proteins — TP (g/dL) [Ref: 6.0–8.3]` | `total_proteins` / `TP` | Total serum protein concentration (albumin + globulins). | Float (`float`) — Streamlit `number_input` (`step=0.1`, default `6.80`) | `g/dL` | **Required** | Min `1.0`, Max `20.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **9** | `Albumin — ALB (g/dL) [Ref: 3.5–5.0]` | `albumin` / `ALB` | Serum albumin protein concentration (hepatic synthetic function marker). | Float (`float`) — Streamlit `number_input` (`step=0.1`, default `3.30`) | `g/dL` | **Required** | Min `0.5`, Max `12.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |
| **10** | `A/G Ratio (dimensionless) [Ref: 1.0–2.5]` | `albumin_and_globulin_ratio` / `A/G Ratio` | Ratio of serum albumin to serum globulin. | Float (`float`) — Streamlit `number_input` (`step=0.05`, default `0.95`) | Ratio (dimensionless) | **Required** | Min `0.1`, Max `10.0` (inclusive). Must be finite numeric; rejects booleans, `NaN`, `Inf`, and non-numeric strings. |

### Field-by-Field User Entry Guidance

1. **Age (`age` / `Age`)**: Enter the patient's age as a whole integer between `1` and `120` years (e.g., `62`). Entered via numeric input box.
2. **Gender (`gender` / `Gender`)**: Select `"Male"` or `"Female"` from the dropdown menu. No other gender strings are accepted by the API.
3. **Total Bilirubin (`total_bilirubin` / `TB`)**: Enter total serum bilirubin in `mg/dL` (normal reference shown in UI: `0.1–1.2 mg/dL`; allowed: `0.05–100.0`).
4. **Direct Bilirubin (`direct_bilirubin` / `DB`)**: Enter conjugated/direct serum bilirubin in `mg/dL` (normal reference shown in UI: `0.0–0.3 mg/dL`; allowed: `0.0–50.0`).
5. **Alkaline Phosphotase (`alkaline_phosphotase` / `Alkphos`)**: Enter serum ALP in `IU/L` (normal reference shown in UI: `44–147 IU/L`; allowed: `10.0–5000.0`). *Note the exact spelling `alkaline_phosphotase` (with an `o`) preserved from the UCI ILPD dataset.*
6. **Alamine Aminotransferase (`alamine_aminotransferase` / `Sgpt`)**: Enter serum ALT/SGPT in `IU/L` (normal reference shown in UI: `7–56 IU/L`; allowed: `1.0–5000.0`). *Note the exact spelling `alamine_aminotransferase` (with an `m`) preserved from the UCI ILPD dataset.*
7. **Aspartate Aminotransferase (`aspartate_aminotransferase` / `Sgot`)**: Enter serum AST/SGOT in `IU/L` (normal reference shown in UI: `10–40 IU/L`; allowed: `1.0–10000.0`).
8. **Total Proteins (`total_proteins` / `TP`)**: Enter total serum protein in `g/dL` (normal reference shown in UI: `6.0–8.3 g/dL`; allowed: `1.0–20.0`).
9. **Albumin (`albumin` / `ALB`)**: Enter serum albumin in `g/dL` (normal reference shown in UI: `3.5–5.0 g/dL`; allowed: `0.5–12.0`).
10. **A/G Ratio (`albumin_and_globulin_ratio` / `A/G Ratio`)**: Enter the albumin-to-globulin ratio as a dimensionless decimal (normal reference shown in UI: `1.0–2.5`; allowed: `0.1–10.0`). This field is **not** calculated automatically from `TP` and `ALB` — the user must enter it explicitly.

### Operational UI Controls (Not Model Inputs)

In addition to the 10 clinical patient fields above, the Streamlit interface (`frontend/app.py`) includes three operational controls:

| Control | Location | Type | Purpose |
| :--- | :--- | :--- | :--- |
| `FastAPI Base URL` | Sidebar | Text input (default `http://localhost:8000` or `API_BASE_URL` env var) | Target URL of the FastAPI inference server. |
| `API Key (X-API-Key)` | Sidebar | Password input (default from `LIVER_API_KEY` env var) | **Required to authenticate requests** to `/predict`, `/explain`, `/predict/batch`, and `/model-info` (missing/wrong key returns HTTP 401). |
| `Include SHAP Feature Attribution Explanation` | Main form | Checkbox (default `True`) | Optional UI toggle: when checked, submits to `POST /explain`; when unchecked, submits to `POST /predict`. Does not alter required patient inputs. |

---

## 2. Number of Required Inputs

1. **How many fields does the user need to fill in?**
   - **Exactly 10 patient input fields** (2 demographic fields + 8 liver function blood test values). *(In the Streamlit UI, the user must also have a valid `X-API-Key` entered in the sidebar if `LIVER_API_KEY` is not already configured in the frontend environment.)*

2. **What are their exact names?**
   - **Frontend labels (`frontend/app.py`)**: `Age (years)`, `Gender`, `Total Bilirubin — TB (mg/dL) [Ref: 0.1–1.2]`, `Direct Bilirubin — DB (mg/dL) [Ref: 0.0–0.3]`, `Alkaline Phosphotase — ALP (IU/L) [Ref: 44–147]`, `Alamine Aminotransferase — ALT/SGPT (IU/L) [Ref: 7–56]`, `Aspartate Aminotransferase — AST/SGOT (IU/L) [Ref: 10–40]`, `Total Proteins — TP (g/dL) [Ref: 6.0–8.3]`, `Albumin — ALB (g/dL) [Ref: 3.5–5.0]`, `A/G Ratio (dimensionless) [Ref: 1.0–2.5]`.
   - **Canonical API JSON / CSV field names (`schemas.py` & `config.py`)**: `age`, `gender`, `total_bilirubin`, `direct_bilirubin`, `alkaline_phosphotase`, `alamine_aminotransferase`, `aspartate_aminotransferase`, `total_proteins`, `albumin`, `albumin_and_globulin_ratio`.
   - **Raw ILPD CSV / internal pipeline names (`config.py` & `liver_disease_pipeline.joblib`)**: `Age`, `Gender`, `TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`, `TP`, `ALB`, `A/G Ratio`.

3. **Which fields are mandatory?**
   - **All 10 patient input fields are strictly mandatory.** In `PatientFeatures`, every field is declared with `Field(...)` (no defaults, no `Optional`) and `ConfigDict(extra="forbid")`.

4. **Which fields are optional?**
   - **Zero patient input fields are optional.** None of the 10 clinical features may be omitted or left null. (Only the UI checkbox `Include SHAP Feature Attribution Explanation` is optional.)

5. **Are there fields that the system calculates automatically?**
   - **Derived clinical ratios are NOT added by the trained model:** Although `ClinicalFeatureEngineer` (`src/liver_disease/features/engineering.py`) implements code for 4 derived biomarkers (`AST_ALT_Ratio`, `DB_TB_Ratio`, `Indirect_Bilirubin`, `Globulin`), inspection of `artifacts/models/liver_disease_pipeline.joblib` and `artifacts/preprocessing/preprocessing_schema.json` confirms that the trained model was fitted with **`add_clinical_ratios=False`** (chosen by the Phase 4 ablation benchmark `log1p_only`).
   - **Internal transformations computed automatically on the 10 inputs:**
     - `np.log1p(max(x, 0))` applied in-place to the 5 skewed biomarkers: `TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`.
     - `RobustScaler` scaling `(x - median) / IQR` applied to all 9 numeric features using fitted training-set parameters.
     - `OneHotEncoder(categories=[["Female", "Male"]], drop="if_binary")` encodes `Gender` into a single binary feature `Gender_Male` (`1.0` = Male, `0.0` = Female).
   - **Response outputs computed automatically:** `prediction_id` (UUID4), `probability` (0.0–1.0), `prediction` (`1` or `2`), `prediction_label` (`"Liver Disease"` / `"No Liver Disease"`), `risk_category` (`"Low"` / `"Moderate"` / `"High"`), `decision_threshold` (`0.57`), and optional SHAP attributions (`KernelExplainer`).

6. **Are there dataset features that the user does NOT need to enter?**
   - **Yes — 1 column from the raw CSV is excluded:** `Selector` (the ground-truth target label: `1` = Liver Disease, `2` = No Liver Disease). The raw ILPD dataset has 11 columns (`10` predictors + `1` target). The user must **never** provide `Selector`; including it in a JSON payload or batch CSV triggers `HTTP 422`. All 10 predictor features in the dataset are required.

7. **Does the application ask for Gender? What are the accepted values?**
   - **Yes.** The accepted values are strictly **`"Male"`** and **`"Female"`** (case-sensitive). In Streamlit, this is enforced via a dropdown (`st.selectbox`). In the API, leading/trailing whitespace is stripped before checking `Literal["Male", "Female"]`. Values such as `"male"`, `"female"`, `"M"`, `"F"`, or `"Other"` are rejected with `HTTP 422`.

8. **Does the application ask for Age?**
   - **Yes.** Age is required as a whole integer in `years`, between `1` and `120` inclusive.

9. **Does the application require all liver blood test values?**
   - **Yes.** All **8** liver function blood test values (`TB`, `DB`, `Alkphos`, `Sgpt`, `Sgot`, `TP`, `ALB`, and `A/G Ratio`) are strictly mandatory. Even though `A/G Ratio` is physiologically related to `ALB` and `TP` (and had 4 missing values in the raw ILPD training dataset), the API and frontend do **not** auto-compute or impute it at inference time — the user must enter all 8 blood test values.

---

## 3. User Interface Example

The Streamlit frontend (`frontend/app.py`, Tab 1: **Individual Patient Prediction**) renders two quick-fill preset buttons (`Load Elevated LFT Sample` and `Load Normal-Range Sample`) above a 3-column form (`single_patient_form`) containing all 10 fields pre-populated with default values. Below is the exact structure from the user's perspective, grouped logically with only fields that exist in the actual application:

### Sidebar (API Connection & Auth)
- **FastAPI Base URL**: `[ Text input ]` — Example: `http://localhost:8000`
- **API Key (`X-API-Key`)**: `[ Password input ]` — Example: `••••••••••••` *(required for API authentication)*

### Patient Information
- **Age (years)**: `[ Number input (integer, 1–120, step 1) ]` — Example: `62` *(UI default: `45`)*
- **Gender**: `[ Dropdown select: Male | Female ]` — Example: `Male` *(UI default: `Male`)*

### Liver Function Test Results
- **Total Bilirubin — TB (mg/dL) [Ref: 0.1–1.2]**: `[ Number input (float, 0.05–100.0, step 0.1) ]` — Example: `10.90` *(UI default: `1.20`)*
- **Direct Bilirubin — DB (mg/dL) [Ref: 0.0–0.3]**: `[ Number input (float, 0.0–50.0, step 0.1) ]` — Example: `5.50` *(UI default: `0.40`)*
- **Alkaline Phosphotase — ALP (IU/L) [Ref: 44–147]**: `[ Number input (float, 10.0–5000.0, step 1.0) ]` — Example: `699.0` *(UI default: `210.0`)*
- **Alamine Aminotransferase — ALT/SGPT (IU/L) [Ref: 7–56]**: `[ Number input (float, 1.0–5000.0, step 1.0) ]` — Example: `64.0` *(UI default: `38.0`)*
- **Aspartate Aminotransferase — AST/SGOT (IU/L) [Ref: 10–40]**: `[ Number input (float, 1.0–10000.0, step 1.0) ]` — Example: `100.0` *(UI default: `45.0`)*
- **Total Proteins — TP (g/dL) [Ref: 6.0–8.3]**: `[ Number input (float, 1.0–20.0, step 0.1) ]` — Example: `7.50` *(UI default: `6.80`)*
- **Albumin — ALB (g/dL) [Ref: 3.5–5.0]**: `[ Number input (float, 0.5–12.0, step 0.1) ]` — Example: `3.20` *(UI default: `3.30`)*
- **A/G Ratio (dimensionless) [Ref: 1.0–2.5]**: `[ Number input (float, 0.1–10.0, step 0.05) ]` — Example: `0.74` *(UI default: `0.95`)*

### Form Controls
- **Include SHAP Feature Attribution Explanation**: `[ Checkbox ]` — Default: `Checked (True)`
- **Action Button**: `[ Run Prediction ]`

*(In the actual Streamlit 3-column layout: Column 1 holds `Age`, `Gender`, `TB`, `DB`; Column 2 holds `Alkphos`, `Sgpt`, `Sgot`; Column 3 holds `TP`, `ALB`, `A/G Ratio`.)*

---

## 4. Complete Example Input

This end-to-end walkthrough uses the **"Elevated LFT Sample"** preset defined in `frontend/app.py` (lines 177–188) and `schemas.py` (lines 36–47), verified directly against the loaded `artifacts/models/liver_disease_pipeline.joblib` artifact.

### 4.1 Values Entered by the User (Raw User Input)

| Frontend Label | Value Entered | Unit |
| :--- | :---: | :--- |
| Age (years) | `62` | years |
| Gender | `Male` | categorical |
| Total Bilirubin — TB | `10.9` | mg/dL |
| Direct Bilirubin — DB | `5.5` | mg/dL |
| Alkaline Phosphotase — ALP | `699.0` | IU/L |
| Alamine Aminotransferase — ALT/SGPT | `64.0` | IU/L |
| Aspartate Aminotransferase — AST/SGOT | `100.0` | IU/L |
| Total Proteins — TP | `7.5` | g/dL |
| Albumin — ALB | `3.2` | g/dL |
| A/G Ratio | `0.74` | dimensionless |

### 4.2 Exact JSON Request Sent to the API

Sent as `POST /predict` (or `POST /explain` when SHAP is enabled) with headers `Content-Type: application/json` and `X-API-Key: <LIVER_API_KEY>`:

```json
{
  "age": 62,
  "gender": "Male",
  "total_bilirubin": 10.9,
  "direct_bilirubin": 5.5,
  "alkaline_phosphotase": 699.0,
  "alamine_aminotransferase": 64.0,
  "aspartate_aminotransferase": 100.0,
  "total_proteins": 7.5,
  "albumin": 3.2,
  "albumin_and_globulin_ratio": 0.74
}
```

### 4.3 Corresponding Model Input After Preprocessing (Verified from `liver_disease_pipeline.joblib`)

The serialized pipeline transforms the 10 user-entered inputs into a 10-element numeric vector in **exact classifier feature order** (`['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Gender_Male']`). Notice how **user-entered values** differ sharply from **transformed model features**:

| Output Index | Transformed Model Feature | User-Entered Value | Step 1: `ClinicalFeatureEngineer` (`log1p` on 5 skewed labs) | Fitted `RobustScaler` Center (Median) | Fitted `RobustScaler` Scale (IQR) | Step 2: Final Transformed Model Input |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0` | `Age` | `62` | `62.000000` *(unchanged)* | `45.000000` | `25.000000` | **`0.680000`** |
| `1` | `TB` | `10.9` | `2.476538` *(`ln(1 + 10.9)`)* | `0.693147` | `0.613599` | **`2.906444`** |
| `2` | `DB` | `5.5` | `1.871802` *(`ln(1 + 5.5)`)* | `0.262364` | `0.606136` | **`2.655243`** |
| `3` | `Alkphos` | `699.0` | `6.551080` *(`ln(1 + 699)`)* | `5.337527` | `0.508537` | **`2.386361`** |
| `4` | `Sgpt` | `64.0` | `4.174387` *(`ln(1 + 64)`)* | `3.583519` | `0.899484` | **`0.656897`** |
| `5` | `Sgot` | `100.0` | `4.615121` *(`ln(1 + 100)`)* | `3.749435` | `1.207812` | **`0.716739`** |
| `6` | `TP` | `7.5` | `7.500000` *(unchanged)* | `6.600000` | `1.300000` | **`0.692308`** |
| `7` | `ALB` | `3.2` | `3.200000` *(unchanged)* | `3.100000` | `1.200000` | **`0.083333`** |
| `8` | `A/G Ratio` | `0.74` | `0.740000` *(unchanged)* | `0.965000` | `0.400000` | **`-0.562500`** |
| `9` | `Gender_Male` | `"Male"` | `"Male"` *(unchanged)* | — *(OneHotEncoder)* | — *(drop=`Female`)* | **`1.000000`** |

**Exact 10-element vector fed to `VotingClassifier` (`ExtraTrees` [w=2] + `RandomForest` [w=1] + `LogisticRegression` [w=1]):**

```text
[0.680000, 2.906444, 2.655243, 2.386361, 0.656897, 0.716739, 0.692308, 0.083333, -0.562500, 1.000000]
```

### 4.4 Verified Prediction Output

Passing the exact input above through `LiverDiseasePredictor.predict_one()` produces:

- **`prediction`**: `1`
- **`prediction_label`**: `"Liver Disease"`
- **`probability`**: `0.9823` *(98.23% statistical model risk score)*
- **`risk_category`**: `"High"` *(>= 0.65)*
- **`decision_threshold`**: `0.57`
- **`model_version`**: `"1.0.0"`

---

## 5. CSV Input (Batch Prediction)

Batch prediction **is supported** via `POST /predict/batch` (`multipart/form-data`, file field name `file`) and **Tab 2: Batch CSV Prediction** in the Streamlit app.

### 5.1 Required CSV Column Names

The uploaded CSV must use **either** of the following **exact 10-column header schemas** (column order does not matter; leading/trailing whitespace around column names is stripped automatically):

1. **Schema A — Canonical API Names (snake_case):**
   ```text
   age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio
   ```
2. **Schema B — Raw UCI ILPD Dataset Names:**
   ```text
   Age,Gender,TB,DB,Alkphos,Sgpt,Sgot,TP,ALB,A/G Ratio
   ```

> **Important:** You cannot mix Schema A and Schema B headers in the same CSV file (e.g., `Age` together with `total_bilirubin` is rejected with `HTTP 422`).

### 5.2 Example CSV Files (Realistic Sample Data)

**Option A — Canonical API Columns** *(matches the downloadable template in `frontend/app.py`)*:
```csv
age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio
62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74
29,Female,0.7,0.1,162,20,21,7.0,3.8,1.18
46,Male,1.8,0.7,208,19,14,7.6,4.4,1.30
```

**Option B — Raw ILPD Columns**:
```csv
Age,Gender,TB,DB,Alkphos,Sgpt,Sgot,TP,ALB,A/G Ratio
62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74
29,Female,0.7,0.1,162,20,21,7.0,3.8,1.18
46,Male,1.8,0.7,208,19,14,7.6,4.4,1.30
```

### 5.3 Mandatory Columns & Extra Columns Policy

- **Mandatory columns:** **All 10 columns** in whichever schema is chosen are mandatory. Missing any column rejects the entire CSV with `HTTP 422`.
- **Extra columns:** **NOT accepted.** The route checks `col_set == raw_set` or `col_set == api_set` strictly (`src/liver_disease/api/routes.py`, lines 237–255). Including any additional column — such as `patient_id`, `name`, `notes`, or the ILPD target column `Selector` — rejects the entire upload with `HTTP 422` listing both missing and unexpected columns.

### 5.4 File Constraints & Handling of Missing or Invalid Values

- **File-level checks:**
  - Filename must end with `.csv` (case-insensitive) → otherwise `HTTP 400`.
  - File must be non-empty, valid UTF-8 (`utf-8-sig`, BOM supported), valid CSV syntax, and contain at least 1 data row → otherwise `HTTP 400`.
  - Row count must not exceed `MAX_BATCH_ROWS` (default **`5000`**, configurable via environment variable) → otherwise `HTTP 413 Payload Too Large`.
- **Row-level validation (`routes.py`, lines 257–286):**
  - Every row is converted to a dictionary (pandas `NaN` values become `None`) and validated individually against `PatientFeatures`.
  - **Missing values are NOT imputed during batch inference.** Any row with a blank/`NaN` cell, invalid gender, non-numeric text, boolean, fractional age, or out-of-range value fails validation and is added to `invalid_records` with its 0-based `row_index` and exact field-level error messages.
  - **Partial success behavior:** If at least one row is valid, the API returns **`HTTP 200`** with `predictions` for all valid rows, `invalid_records` detailing all skipped rows, and summary counts (`total_rows`, `valid_count`, `invalid_count`).
  - **Total failure behavior:** If **100% of rows** fail validation, the API returns **`HTTP 422`** with the message `"All rows in the uploaded CSV failed medical input validation."` and the first 25 `invalid_records`.

---

## 6. Final Integration Explanation (End-to-End Data Flow)

```text
User Input (10 clinical fields + X-API-Key)
   │
   ▼
1. Frontend Form (frontend/app.py — Streamlit)
   • Collects 10 fields with client-side min/max bounds & dropdowns (or CSV upload)
   • Builds snake_case JSON payload (or multipart/form-data file)
   • Attaches X-API-Key header and sends HTTP POST to /predict, /explain, or /predict/batch
   │
   ▼
2. FastAPI Authentication & Routing (api/security.py, api/routes.py)
   • Constant-time HMAC verification of X-API-Key against LIVER_API_KEY env var (401/503 on failure)
   • Retrieves pre-loaded LiverDiseasePredictor from app.state (never retrains at startup)
   │
   ▼
3. Input Validation (api/schemas.py + data/loader.py + data/validation.py)
   • Layer 1 (Pydantic v2 PatientFeatures): rejects missing fields, extra fields (extra="forbid"),
     booleans, NaN/Inf, fractional age, invalid gender, and out-of-range numeric values (HTTP 422)
   • Layer 2 (normalize_inference_columns + validate_feature_dataframe): maps snake_case API keys
     to exact 10 raw ILPD columns in deterministic order and re-verifies bounds with allow_missing=False
   │
   ▼
4. Preprocessing Pipeline (serialized inside liver_disease_pipeline.joblib)
   • ClinicalFeatureEngineer(add_clinical_ratios=False, log_transform_skewed=True):
     applies np.log1p(max(x, 0)) in-place to [TB, DB, Alkphos, Sgpt, Sgot]
   • ColumnTransformer:
     - Numeric branch (9 cols): SimpleImputer(median) → RobustScaler(center=median, scale=IQR)
     - Categorical branch (Gender): SimpleImputer(most_frequent) → OneHotEncoder(drop="if_binary") → Gender_Male
   • Produces 10-element float array: [Age, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio, Gender_Male]
   │
   ▼
5. ML Model Inference (SoftVoting_ET_RF_LR VotingClassifier, weights [2, 1, 1])
   • Combines predicted probabilities from ExtraTreesClassifier (w=2), RandomForestClassifier (w=1),
     and LogisticRegression (w=1); calibration_method = "none"
   │
   ▼
6. Prediction Post-Processing & Optional SHAP Explainability (models/predict.py, models/explain.py)
   • Clips probability P(Liver Disease) to [0, 1] and rounds to 4 decimals
   • Applies validation-tuned decision threshold 0.57:
     prob >= 0.57 → prediction = 1 ("Liver Disease"); prob < 0.57 → prediction = 2 ("No Liver Disease")
   • Maps probability to Risk Category: Low (<0.35), Moderate (0.35–<0.65), High (>=0.65)
   • For /explain: runs SHAP KernelExplainer against saved training background (top 6 + all 10 features)
   │
   ▼
7. Result Display (frontend/app.py)
   • Renders 4 KPI metric cards (Class, Probability %, Risk Category, Decision Threshold),
     Plotly probability gauge with threshold marker, and horizontal SHAP attribution bar chart
   • For batch CSV: displays summary KPIs, predictions table, invalid rows warning table,
     and downloadable merged predictions CSV
```

---

## 7. Critical Verification: Frontend vs. API vs. Trained Model

### 7.1 Cross-Layer Alignment Matrix

| # | Frontend Input (`frontend/app.py`) | Frontend Bounds | API Schema (`schemas.py`) | API Bounds | Trained Model Raw Input (`feature_engineer.feature_names_in_`) | Trained Model Final Feature (`classifier`, `n_features_in_=10`) | Alignment Status |
| :---: | :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | `Age (years)` | `1 – 120` (`int`) | `age` | `1 – 120` (`int`) | `Age` | `Age` (idx 0, scaled) | **Exact Match** |
| 2 | `Gender` | `Male` / `Female` | `gender` | `Male` / `Female` | `Gender` | `Gender_Male` (idx 9, 0/1) | **Exact Match** |
| 3 | `Total Bilirubin — TB` | `0.05 – 100.0` | `total_bilirubin` | `0.05 – 100.0` | `TB` | `TB` (idx 1, log1p + scaled) | **Exact Match** |
| 4 | `Direct Bilirubin — DB` | `0.0 – 50.0` | `direct_bilirubin` | `0.0 – 50.0` | `DB` | `DB` (idx 2, log1p + scaled) | **Exact Match** |
| 5 | `Alkaline Phosphotase — ALP` | `10.0 – 5000.0` | `alkaline_phosphotase` | `10.0 – 5000.0` | `Alkphos` | `Alkphos` (idx 3, log1p + scaled) | **Exact Match** |
| 6 | `Alamine Aminotransferase — ALT/SGPT` | `1.0 – 5000.0` | `alamine_aminotransferase` | `1.0 – 5000.0` | `Sgpt` | `Sgpt` (idx 4, log1p + scaled) | **Exact Match** |
| 7 | `Aspartate Aminotransferase — AST/SGOT` | `1.0 – 10000.0` | `aspartate_aminotransferase` | `1.0 – 10000.0` | `Sgot` | `Sgot` (idx 5, log1p + scaled) | **Exact Match** |
| 8 | `Total Proteins — TP` | `1.0 – 20.0` | `total_proteins` | `1.0 – 20.0` | `TP` | `TP` (idx 6, scaled) | **Exact Match** |
| 9 | `Albumin — ALB` | `0.5 – 12.0` | `albumin` | `0.5 – 12.0` | `ALB` | `ALB` (idx 7, scaled) | **Exact Match** |
| 10 | `A/G Ratio` | `0.1 – 10.0` | `albumin_and_globulin_ratio` | `0.1 – 10.0` | `A/G Ratio` | `A/G Ratio` (idx 8, scaled) | **Exact Match** |

### 7.2 Verification Verdict & Important Implementation Notes

- **Verdict:** **All three layers (Streamlit frontend, FastAPI/Pydantic API, and the trained `SoftVoting_ET_RF_LR` pipeline artifact) are 100% synchronized.** There are **zero** missing fields, extra fields, bound mismatches, or broken mappings between the frontend, API, and model.
- **Six Verified Caveats & Integration Notes to Keep in Mind:**
  1. **Dataset Legacy Spellings Are Enforced Strictly:** The API keys and CSV headers use `alkaline_phosphotase` (not `alkaline_phosphatase`) and `alamine_aminotransferase` (not `alanine_aminotransferase`). While all internal layers agree on these spellings, external callers who use standard medical spelling will receive `HTTP 422`.
  2. **Dormant Clinical Ratio Code (`add_clinical_ratios=False`):** `ClinicalFeatureEngineer` defaults to `add_clinical_ratios=True` in its class definition, but `train.py` and `scripts/run_training.py` trained the serialized production model with `add_clinical_ratios=False`. Because the fitted transformer is serialized inside `liver_disease_pipeline.joblib`, inference cleanly produces 10 features (not 14) with no mismatch.
  3. **Imputers Inside Pipeline Are Unreachable via API:** Both numeric (`median`) and categorical (`most_frequent`) `SimpleImputer` steps exist inside the serialized pipeline (necessary during training because raw ILPD had 4 missing `A/G Ratio` values), but the REST API enforces zero missing values (`PatientFeatures` + `allow_missing=False`).
  4. **No Cross-Field Physiological Validation at Inference:** While per-field min/max bounds are enforced strictly, cross-field clinical inequalities (`direct_bilirubin <= total_bilirubin`, `albumin <= total_proteins`, or `albumin_and_globulin_ratio ≈ albumin / (total_proteins - albumin)`) are **not** rejected by the frontend or API (consistent with the raw ILPD dataset, which contains 3 rows with `DB > TB`).
  5. **Minor Stale Illustrative Value in `API_DOCUMENTATION.md`:** In `API_DOCUMENTATION.md` (line 159), an abbreviated example `/explain` snippet shows `DB` `transformed_value: 1.4481`, whereas the actual serialized model computes `2.6552` for `DB = 5.5` (and `base_value: 0.71264` vs `0.70614`). This does not affect runtime execution; Section 4 of this guide reports the verified live model values.
  6. **Strict Batch CSV Export Merging Assumption:** In `frontend/app.py` (lines 345–347), the Streamlit batch download merges API predictions back onto the uploaded CSV via 0-based `row_index`; invalid rows appear in the exported CSV with blank prediction columns alongside the warning table displayed in the UI.

---

## Direct Final Answer

**To get a prediction, the user must enter exactly 10 fields:** `age` (`Age`), `gender` (`Gender`), `total_bilirubin` (`TB`), `direct_bilirubin` (`DB`), `alkaline_phosphotase` (`Alkphos`), `alamine_aminotransferase` (`Sgpt`), `aspartate_aminotransferase` (`Sgot`), `total_proteins` (`TP`), `albumin` (`ALB`), and `albumin_and_globulin_ratio` (`A/G Ratio`).
