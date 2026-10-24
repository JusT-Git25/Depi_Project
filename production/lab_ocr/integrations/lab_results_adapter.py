from dataclasses import dataclass
from ..schemas import LabReportResult

_INEXACT_OPERATORS = {"<", "<=", ">", ">="}


def _norm_unit(unit: str | None) -> str:
    if not unit:
        return ""
    return "".join(unit.split()).lower().replace("µ", "u").replace("μ", "u")


@dataclass
class PredictionInput:
    features: dict[str, float | str]
    warnings: list[str]


def prepare_prediction_features(
    report: LabReportResult,
    feature_mapping: dict[str, str],
    *,
    allow_unreviewed: bool = False,
    expected_units: dict[str, str] | None = None,
    exact_values_only: bool = False,
) -> PredictionInput:
    """Map reviewed laboratory results to model feature names without calling a model.

    feature_mapping maps canonical OCR test_name -> exact model feature name.
    expected_units (optional) maps model feature name -> the unit the model was trained on.
    A result whose unit is missing or different is excluded (no unit conversion is done).
    exact_values_only excludes values such as "<0.5" that are not exact measurements.
    """
    expected_units = expected_units or {}
    features: dict[str, float | str] = {}
    warnings: list[str] = []
    for result in report.results:
        if result.test_name not in feature_mapping:
            continue
        if not result.manually_reviewed and not allow_unreviewed:
            warnings.append(f"{result.test_name}: excluded because it has not been manually reviewed.")
            continue
        if result.validation_status == "invalid":
            warnings.append(f"{result.test_name}: excluded because validation status is invalid.")
            continue
        if result.value is None:
            warnings.append(f"{result.test_name}: excluded because no value was parsed.")
            continue
        if exact_values_only and result.value_operator in _INEXACT_OPERATORS:
            warnings.append(f"{result.test_name}: excluded because the value is not exact ({result.value_operator}).")
            continue
        target = feature_mapping[result.test_name]
        wanted_unit = expected_units.get(target)
        if wanted_unit and _norm_unit(result.unit) != _norm_unit(wanted_unit):
            found = result.unit or "no unit"
            warnings.append(
                f"{result.test_name}: excluded because the unit is '{found}' but the model expects '{wanted_unit}'. "
                "Units are never converted automatically."
            )
            continue
        if target in features:
            warnings.append(f"{result.test_name}: duplicate candidate for feature '{target}'; skipped.")
            continue
        features[target] = result.value
    return PredictionInput(features=features, warnings=warnings)
