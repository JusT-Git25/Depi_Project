"""Central configuration and single source of truth for the Liver Disease Prediction System."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Project root resolution
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Canonical directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
PREPROCESSING_DIR: Path = ARTIFACTS_DIR / "preprocessing"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Dataset filenames
RAW_DATASET_FILENAME: str = "Indian Liver Patient Dataset (ILPD).csv"
RAW_DATASET_PATH: Path = RAW_DATA_DIR / RAW_DATASET_FILENAME
ROOT_DATASET_FALLBACK: Path = PROJECT_ROOT / RAW_DATASET_FILENAME

# Artifact paths
DEFAULT_MODEL_ARTIFACT_PATH: Path = MODELS_DIR / "liver_disease_pipeline.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# Reproducibility & modeling constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
MODEL_VERSION: str = "1.0.0"

# Target definitions (Original ILPD encoding)
TARGET_COLUMN: str = "Selector"
TARGET_DISEASE_RAW: int = 1       # 1 = Liver Disease (Positive class -> binary 1)
TARGET_NO_DISEASE_RAW: int = 2    # 2 = No Liver Disease (Negative class -> binary 0)
TARGET_LABELS: Dict[int, str] = {
    1: "Liver Disease",
    2: "No Liver Disease",
}

# Raw ILPD feature column order
RAW_FEATURE_COLUMNS: List[str] = [
    "Age",
    "Gender",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]

NUMERIC_FEATURE_COLUMNS: List[str] = [
    "Age",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]

CATEGORICAL_FEATURE_COLUMNS: List[str] = ["Gender"]
VALID_GENDERS: Tuple[str, ...] = ("Male", "Female")

# Mapping between canonical API snake_case names and raw ILPD CSV column names
API_TO_RAW_COLUMN_MAP: Dict[str, str] = {
    "age": "Age",
    "gender": "Gender",
    "total_bilirubin": "TB",
    "direct_bilirubin": "DB",
    "alkaline_phosphotase": "Alkphos",
    "alamine_aminotransferase": "Sgpt",
    "aspartate_aminotransferase": "Sgot",
    "total_proteins": "TP",
    "albumin": "ALB",
    "albumin_and_globulin_ratio": "A/G Ratio",
}

RAW_TO_API_COLUMN_MAP: Dict[str, str] = {v: k for k, v in API_TO_RAW_COLUMN_MAP.items()}


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for a single ILPD predictor feature."""

    raw_name: str
    api_name: str
    display_name: str
    dtype: str
    unit: str
    min_value: float | None = None
    max_value: float | None = None
    normal_reference_range: str = ""
    description: str = ""


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "Age": FeatureSpec(
        raw_name="Age",
        api_name="age",
        display_name="Age",
        dtype="int",
        unit="years",
        min_value=1.0,
        max_value=120.0,
        normal_reference_range="1–120 years",
        description="Age of the patient in years (patients >89 are recorded as 90 in UCI ILPD).",
    ),
    "Gender": FeatureSpec(
        raw_name="Gender",
        api_name="gender",
        display_name="Gender",
        dtype="category",
        unit="categorical (Male/Female)",
        normal_reference_range="Male or Female",
        description="Biological sex / gender recorded in the dataset.",
    ),
    "TB": FeatureSpec(
        raw_name="TB",
        api_name="total_bilirubin",
        display_name="Total Bilirubin (TB)",
        dtype="float",
        unit="mg/dL",
        min_value=0.05,
        max_value=100.0,
        normal_reference_range="0.1–1.2 mg/dL",
        description="Total serum bilirubin concentration.",
    ),
    "DB": FeatureSpec(
        raw_name="DB",
        api_name="direct_bilirubin",
        display_name="Direct Bilirubin (DB)",
        dtype="float",
        unit="mg/dL",
        min_value=0.0,
        max_value=50.0,
        normal_reference_range="0.0–0.3 mg/dL",
        description="Conjugated (direct) serum bilirubin concentration.",
    ),
    "Alkphos": FeatureSpec(
        raw_name="Alkphos",
        api_name="alkaline_phosphotase",
        display_name="Alkaline Phosphotase (ALP)",
        dtype="float",
        unit="IU/L",
        min_value=10.0,
        max_value=5000.0,
        normal_reference_range="44–147 IU/L",
        description="Serum alkaline phosphatase enzyme level.",
    ),
    "Sgpt": FeatureSpec(
        raw_name="Sgpt",
        api_name="alamine_aminotransferase",
        display_name="Alamine Aminotransferase (ALT / SGPT)",
        dtype="float",
        unit="IU/L",
        min_value=1.0,
        max_value=5000.0,
        normal_reference_range="7–56 IU/L",
        description="Serum alanine aminotransferase enzyme level.",
    ),
    "Sgot": FeatureSpec(
        raw_name="Sgot",
        api_name="aspartate_aminotransferase",
        display_name="Aspartate Aminotransferase (AST / SGOT)",
        dtype="float",
        unit="IU/L",
        min_value=1.0,
        max_value=10000.0,
        normal_reference_range="10–40 IU/L",
        description="Serum aspartate aminotransferase enzyme level.",
    ),
    "TP": FeatureSpec(
        raw_name="TP",
        api_name="total_proteins",
        display_name="Total Proteins (TP)",
        dtype="float",
        unit="g/dL",
        min_value=1.0,
        max_value=20.0,
        normal_reference_range="6.0–8.3 g/dL",
        description="Total serum protein concentration.",
    ),
    "ALB": FeatureSpec(
        raw_name="ALB",
        api_name="albumin",
        display_name="Albumin (ALB)",
        dtype="float",
        unit="g/dL",
        min_value=0.5,
        max_value=12.0,
        normal_reference_range="3.5–5.0 g/dL",
        description="Serum albumin protein concentration.",
    ),
    "A/G Ratio": FeatureSpec(
        raw_name="A/G Ratio",
        api_name="albumin_and_globulin_ratio",
        display_name="Albumin and Globulin Ratio (A/G Ratio)",
        dtype="float",
        unit="ratio (dimensionless)",
        min_value=0.1,
        max_value=10.0,
        normal_reference_range="1.0–2.5",
        description="Ratio of serum albumin to serum globulin.",
    ),
}

# Risk categorization thresholds based on model score P(Liver Disease)
# Clearly distinguished from binary decision threshold and NOT clinically calibrated probabilities.
RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 0.35),
    "Moderate": (0.35, 0.65),
    "High": (0.65, 1.01),
}

MEDICAL_DISCLAIMER: str = (
    "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER: This system is an academic machine learning "
    "prototype trained on the public Indian Liver Patient Dataset (ILPD). It is NOT a "
    "clinically validated medical device and MUST NOT be used to diagnose, treat, or "
    "make clinical decisions for patients. Predicted probabilities represent statistical "
    "model risk scores rather than clinically calibrated diagnostic probabilities."
)


def ensure_project_directories() -> None:
    """Create standard project directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        PREPROCESSING_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
