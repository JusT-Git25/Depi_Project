import logging
import math
from typing import Any, Literal
from pydantic import BaseModel, Field
from ..exceptions import PredictionError
from ..integrations.lab_results_adapter import prepare_prediction_features
from ..schemas import LabReportResult
from .spec import ModelSpec

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is a statistical estimate computed from automatically extracted lab values. "
    "It is not a diagnosis. Confirm any result with a qualified clinician."
)


def _bundle_helpers():
    """Lazy access to the production bridge (kept optional: lab_ocr can run standalone)."""
    try:
        from production.integrations.production_bundle import is_production_bundle, run_bundle
        return is_production_bundle, run_bundle
    except ImportError:
        return None, None


class PredictionResult(BaseModel):
    status: Literal["ok", "insufficient_data"]
    prediction: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    used_features: dict[str, float] = Field(default_factory=dict)
    missing_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_name: str
    model_version: str | None = None
    demo: bool = False
    # Optional fields populated when the production bundle bridge is used.
    risk_score: float | None = None
    risk_percent: str | None = None
    final_level: str | None = None
    risk_level: str | None = None
    flagged_for_screening: bool | None = None
    recommendation: str | None = None
    top_risk_factors: list[dict] = Field(default_factory=list)
    lab_signals: dict[str, float] = Field(default_factory=dict)
    clinical_assessment: dict | None = None
    glycemic_alert: dict | None = None
    survey_risk: dict | None = None
    advice: dict | None = None
    disclaimer: str = DISCLAIMER


def _label_text(value: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


class RiskPredictor:
    """Runs a trained scikit-learn-style model on reviewed lab results.

    The model is never retrained or modified. If required features are missing, no prediction is made.
    """

    def __init__(self, spec: ModelSpec, model: Any | None = None):
        self.spec = spec
        if model is not None:
            if isinstance(model, dict) and "model" in model:
                try:
                    from production.integrations.production_bundle import ProductionRiskModel
                    model = ProductionRiskModel(model)
                except ImportError:
                    pass
            self._model = model
        else:
            self._model = self._load_model(spec.model_path)

    @classmethod
    def from_spec_file(cls, path) -> "RiskPredictor":
        return cls(ModelSpec.load(path))

    @staticmethod
    def _load_model(path: str):
        try:
            import joblib
        except ImportError as exc:
            raise PredictionError("Prediction needs joblib. Install with: pip install -e \".[predict]\"") from exc
        try:
            # joblib uses pickle: only load model files you created or fully trust.
            model = joblib.load(path)
        except Exception as exc:
            raise PredictionError("The model file could not be loaded.") from exc
        # A production bundle is a dict of model + calibrator + metadata; wrap it so
        # the NaN-tolerant, calibrated path in the bridge is used.
        if isinstance(model, dict) and "model" in model:
            try:
                from production.integrations.production_bundle import ProductionRiskModel
                return ProductionRiskModel(model)
            except ImportError:
                raise PredictionError(
                    "This is a production model bundle; run it from the project root so the "
                    "`production` package is importable.")
        return model

    def predict_report(self, report: LabReportResult, *, allow_unreviewed: bool = False) -> PredictionResult:
        spec = self.spec
        prepared = prepare_prediction_features(
            report, spec.feature_mapping,
            allow_unreviewed=allow_unreviewed,
            expected_units=spec.expected_units,
            exact_values_only=True,
        )
        warnings = list(prepared.warnings)
        features: dict[str, float] = {}
        for name, value in prepared.features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                features[name] = float(value)
            else:
                warnings.append(f"{name}: excluded because the value is not numeric.")

        is_bundle, run_bundle = _bundle_helpers()
        if is_bundle and is_bundle(self._model):
            return self._predict_bundle(spec, features, warnings, allow_unreviewed, run_bundle)

        filled: dict[str, float] = {}
        for name in spec.feature_order:
            if name not in features and name in spec.fill_values:
                filled[name] = spec.fill_values[name]
                warnings.append(f"{name}: not found in the report; the configured default {filled[name]} was used.")
        missing = [n for n in spec.feature_order if n not in features and n not in filled]

        base = dict(
            used_features=features, missing_features=missing, warnings=warnings,
            model_name=spec.name, model_version=spec.version, demo=spec.demo,
        )
        if missing:
            return PredictionResult(status="insufficient_data", **base)

        row = {**features, **filled}
        try:
            import pandas as pd
            frame = pd.DataFrame([[row[n] for n in spec.feature_order]], columns=spec.feature_order)
            raw_prediction = self._model.predict(frame)[0]
            probabilities: dict[str, float] = {}
            if hasattr(self._model, "predict_proba"):
                classes = getattr(self._model, "classes_", None)
                if classes is None and hasattr(self._model, "steps"):
                    classes = self._model.steps[-1][1].classes_
                probs = self._model.predict_proba(frame)[0]
                if classes is not None:
                    probabilities = {
                        spec.labels.get(_label_text(c), _label_text(c)): round(float(p), 4)
                        for c, p in zip(classes, probs)
                    }
        except Exception as exc:
            logger.warning("Model execution failed (%s)", type(exc).__name__)
            raise PredictionError("The model could not run on these values. Check the model spec and feature order.") from exc

        key = _label_text(raw_prediction)
        return PredictionResult(status="ok", prediction=spec.labels.get(key, key), probabilities=probabilities, **base)

    def _predict_bundle(self, spec, features: dict[str, float], warnings: list[str],
                        allow_unreviewed: bool, run_bundle) -> PredictionResult:
        """NaN-tolerant path for the production bundle: only the confirmed labs are used.

        Unlike the generic path we do NOT require every feature in `feature_order`:
        the production model is trained to accept missing (survey) fields.
        """
        import pandas as pd

        base = dict(model_name=spec.name, model_version=spec.version, demo=spec.demo)
        if not features:
            warnings.append("No reviewed lab values mapped to the model; no prediction made.")
            return PredictionResult(status="insufficient_data", used_features={},
                                    missing_features=list(spec.feature_order), warnings=warnings, **base)

        frame = pd.DataFrame([features])
        bundle = getattr(self._model, "bundle", self._model)
        try:
            out = run_bundle(bundle, frame, spec.feature_order, allow_unreviewed)
        except Exception as exc:
            logger.warning("Production model execution failed (%s)", type(exc).__name__)
            raise PredictionError("The production model could not run on these values.") from exc

        warnings.extend(w for w in out.get("warnings", []) if w not in warnings)
        if out["status"] != "ok":
            return PredictionResult(
                status="insufficient_data",
                used_features=out.get("used_features", {}),
                missing_features=out.get("missing_features", []),
                lab_signals=out.get("lab_signals", {}),
                clinical_assessment=out.get("clinical_assessment"),
                glycemic_alert=out.get("glycemic_alert"),
                survey_risk=out.get("survey_risk"),
                warnings=warnings,
                **base,
            )
        return PredictionResult(
            status="ok",
            prediction=out["prediction"],
            final_level=out.get("final_level") or out["prediction"],
            risk_level=out.get("risk_level") or out["prediction"],
            probabilities=out["probabilities"],
            used_features=out["used_features"],
            missing_features=out["missing_features"],
            warnings=warnings,
            risk_score=out.get("risk_score"),
            risk_percent=out.get("risk_percent"),
            flagged_for_screening=out.get("flagged_for_screening"),
            recommendation=out.get("recommendation"),
            top_risk_factors=out.get("top_risk_factors", []),
            lab_signals=out.get("lab_signals", {}),
            clinical_assessment=out.get("clinical_assessment"),
            glycemic_alert=out.get("glycemic_alert"),
            survey_risk=out.get("survey_risk"),
            **base,
        )
