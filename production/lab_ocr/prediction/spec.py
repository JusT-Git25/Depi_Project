import json
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from ..exceptions import PredictionError


class ModelSpec(BaseModel):
    """Describes how OCR lab results map onto a trained model's inputs.

    Everything here must be checked against the real model: feature names, order and units.
    """
    name: str = "risk-model"
    version: str | None = None
    demo: bool = False
    model_path: str
    feature_order: list[str] = Field(min_length=1)
    feature_mapping: dict[str, str]          # canonical OCR test_name -> model feature name
    expected_units: dict[str, str] = Field(default_factory=dict)   # model feature -> unit
    fill_values: dict[str, float] = Field(default_factory=dict)    # model feature -> default if missing
    labels: dict[str, str] = Field(default_factory=dict)           # class value (as text) -> display name

    @model_validator(mode="after")
    def _check_consistency(self):
        order = set(self.feature_order)
        if len(order) != len(self.feature_order):
            raise ValueError("feature_order contains duplicates.")
        for target in self.feature_mapping.values():
            if target not in order:
                raise ValueError(f"feature_mapping targets '{target}' which is not in feature_order.")
        for name, mapping in (("expected_units", self.expected_units), ("fill_values", self.fill_values)):
            for key in mapping:
                if key not in order:
                    raise ValueError(f"{name} refers to '{key}' which is not in feature_order.")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ModelSpec":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PredictionError("The model spec file could not be read.") from exc
        try:
            spec = cls.model_validate(data)
        except ValueError as exc:
            raise PredictionError(f"The model spec is invalid: {exc}") from exc
        model_file = Path(spec.model_path)
        if not model_file.is_absolute():
            model_file = (path.parent / model_file).resolve()
        return spec.model_copy(update={"model_path": str(model_file)})
