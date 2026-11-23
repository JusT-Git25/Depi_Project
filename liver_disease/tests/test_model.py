"""Tests for serialized model artifact integrity and loading behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from liver_disease.config import DEFAULT_MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH
from liver_disease.models.evaluate import (
    compute_classification_metrics,
    optimize_validation_threshold,
)
from liver_disease.models.predict import LiverDiseasePredictor
from liver_disease.utils.io import load_json


def test_serialized_model_artifact_loads_cleanly() -> None:
    assert DEFAULT_MODEL_ARTIFACT_PATH.exists(), "Trained model artifact must exist."
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)
    info = predictor.get_model_info()

    assert info["model_version"] == "1.0.0"
    assert info["model_name"]
    assert 0.0 < info["decision_threshold"] < 1.0
    assert info["explainability_supported"] is True
    assert len(info["global_feature_importance"]) == len(info["transformed_features"])
    assert "holdout_test_selected_threshold" in info["evaluation_metrics"]


def test_model_metadata_json_matches_loaded_artifact() -> None:
    assert MODEL_METADATA_PATH.exists()
    meta = load_json(MODEL_METADATA_PATH)
    predictor = LiverDiseasePredictor(DEFAULT_MODEL_ARTIFACT_PATH)
    assert meta["model_version"] == predictor.model_version
    assert meta["model_name"] == predictor.model_name
    assert float(meta["decision_threshold"]) == pytest.approx(predictor.decision_threshold, abs=1e-6)
    assert "dependency_versions" in meta
    assert "trained_at_utc" in meta


def test_missing_artifact_fails_explicitly_without_retraining(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="never retrains automatically"):
        LiverDiseasePredictor(missing_path)


def test_evaluation_metrics_and_threshold_optimizer() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.92, 0.85, 0.72, 0.55, 0.48, 0.30, 0.20, 0.10])
    m = compute_classification_metrics(y_true, y_proba, threshold=0.50)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["recall_sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 4}

    thr, report = optimize_validation_threshold(y_true, y_proba, min_specificity=0.50)
    assert 0.15 <= thr <= 0.85
    assert report["selected_oof_metrics"]["specificity"] >= 0.50
