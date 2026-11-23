"""Production inference engine and reliable artifact loader (never retrains at startup)."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from liver_disease.config import (
    DEFAULT_MODEL_ARTIFACT_PATH,
    MEDICAL_DISCLAIMER,
    RAW_FEATURE_COLUMNS,
    RISK_THRESHOLDS,
    TARGET_DISEASE_RAW,
    TARGET_LABELS,
    TARGET_NO_DISEASE_RAW,
)
from liver_disease.data.loader import normalize_inference_columns
from liver_disease.data.validation import validate_feature_dataframe
from liver_disease.models.explain import build_shap_explainer, explain_single_record

logger = logging.getLogger(__name__)


def categorize_risk(probability: float, risk_thresholds: Dict[str, tuple[float, float]] | None = None) -> str:
    """Map model probability score P(Liver Disease) to a qualitative risk tier.

    Note: Risk tiers represent model score bands, NOT clinically validated diagnostic risk.
    """
    tiers = risk_thresholds or RISK_THRESHOLDS
    low_upper = tiers["Low"][1]
    mod_upper = tiers["Moderate"][1]
    if probability < low_upper:
        return "Low"
    if probability < mod_upper:
        return "Moderate"
    return "High"


class LiverDiseasePredictor:
    """Thread-safe inference wrapper loaded from a versioned joblib artifact bundle."""

    def __init__(self, artifact_path: Path | str | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path else DEFAULT_MODEL_ARTIFACT_PATH
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.artifact_path}'. "
                "Run `python scripts/run_training.py` to train and serialize the pipeline first. "
                "The API intentionally never retrains automatically on startup."
            )

        bundle: Dict[str, Any] = joblib.load(self.artifact_path)
        required_keys = {
            "model_version",
            "model_name",
            "model_type",
            "pipeline",
            "decision_threshold",
            "transformed_feature_names",
        }
        missing = required_keys - set(bundle.keys())
        if missing:
            raise ValueError(f"Corrupted model artifact bundle; missing keys: {sorted(missing)}")

        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.explain_pipeline = bundle.get("explain_pipeline", self.pipeline)
        self.model_version: str = str(bundle["model_version"])
        self.model_name: str = str(bundle["model_name"])
        self.model_type: str = str(bundle["model_type"])
        self.decision_threshold: float = float(bundle["decision_threshold"])
        self.default_threshold: float = float(bundle.get("default_threshold", 0.50))
        self.risk_thresholds: Dict[str, tuple[float, float]] = bundle.get("risk_thresholds", RISK_THRESHOLDS)
        self.transformed_feature_names: List[str] = list(bundle["transformed_feature_names"])
        self.shap_background: np.ndarray | None = bundle.get("shap_background")

        # Initialize SHAP explainer once at load time if background is present
        self._shap_explainer: Any | None = None
        self._shap_explainer_type: str | None = None
        if self.shap_background is not None:
            try:
                self._shap_explainer, self._shap_explainer_type = build_shap_explainer(
                    self.explain_pipeline,
                    np.asarray(self.shap_background, dtype=float),
                )
            except Exception as exc:
                logger.warning("Could not pre-initialize SHAP explainer: %s", exc)

        logger.info(
            "Loaded model artifact v%s (%s, threshold=%.2f) from %s",
            self.model_version,
            self.model_name,
            self.decision_threshold,
            self.artifact_path,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return safe, structured model metadata for /model-info endpoint."""
        return {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "calibration_method": self.bundle.get("calibration_method", "none"),
            "decision_threshold": self.decision_threshold,
            "default_threshold": self.default_threshold,
            "risk_category_bands": {
                "Low": f"< {self.risk_thresholds['Low'][1]:.2f}",
                "Moderate": f"{self.risk_thresholds['Moderate'][0]:.2f} – < {self.risk_thresholds['Moderate'][1]:.2f}",
                "High": f">= {self.risk_thresholds['High'][0]:.2f}",
            },
            "target_encoding": {
                "1": TARGET_LABELS[1],
                "2": TARGET_LABELS[2],
            },
            "required_features_raw": RAW_FEATURE_COLUMNS,
            "transformed_features": self.transformed_feature_names,
            "explainability_supported": self.shap_background is not None,
            "explainer_type": self._shap_explainer_type,
            "trained_at_utc": self.bundle.get("trained_at_utc"),
            "dataset_info": self.bundle.get("dataset_info", {}),
            "evaluation_metrics": self.bundle.get("evaluation_metrics", {}),
            "global_feature_importance": self.bundle.get("global_feature_importance", []),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def predict_dataframe(self, df: pd.DataFrame, allow_missing: bool = False) -> List[Dict[str, Any]]:
        """Run validated batch inference on a DataFrame of patient records."""
        normalized = normalize_inference_columns(df)
        errors = validate_feature_dataframe(normalized, allow_missing=allow_missing)
        if errors:
            raise ValueError("Input validation failed: " + " | ".join(errors))

        # Normalize Gender capitalization cleanly
        normalized["Gender"] = normalized["Gender"].astype(str).str.strip()
        probas = self.pipeline.predict_proba(normalized)[:, 1]

        results: List[Dict[str, Any]] = []
        for idx, prob in enumerate(probas):
            prob_f = float(np.clip(prob, 0.0, 1.0))
            is_disease = prob_f >= self.decision_threshold
            pred_class = TARGET_DISEASE_RAW if is_disease else TARGET_NO_DISEASE_RAW
            results.append(
                {
                    "row_index": idx,
                    "prediction_id": str(uuid.uuid4()),
                    "prediction": pred_class,
                    "prediction_label": TARGET_LABELS[pred_class],
                    "probability": round(prob_f, 4),
                    "risk_category": categorize_risk(prob_f, self.risk_thresholds),
                    "decision_threshold": self.decision_threshold,
                    "model_version": self.model_version,
                }
            )
        return results

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict liver disease likelihood for a single patient dictionary."""
        df = pd.DataFrame([features])
        result = self.predict_dataframe(df, allow_missing=False)[0]
        result.pop("row_index", None)
        result["disclaimer"] = MEDICAL_DISCLAIMER
        return result

    def explain_one(self, features: Dict[str, Any], top_k: int = 6) -> Dict[str, Any]:
        """Return both prediction and SHAP explanation for a single patient."""
        if self.shap_background is None:
            raise RuntimeError("SHAP explainability background data is not available in this model artifact.")

        df = normalize_inference_columns(pd.DataFrame([features]))
        errors = validate_feature_dataframe(df, allow_missing=False)
        if errors:
            raise ValueError("Input validation failed: " + " | ".join(errors))

        pred = self.predict_one(features)
        explanation = explain_single_record(
            pipeline=self.explain_pipeline,
            record_df=df,
            feature_names=self.transformed_feature_names,
            X_background_transformed=np.asarray(self.shap_background, dtype=float),
            explainer=self._shap_explainer,
            explainer_type=self._shap_explainer_type,
            top_k=top_k,
        )
        return {
            **pred,
            "explanation": explanation,
        }
