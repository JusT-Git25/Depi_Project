"""Cross-module integration layer (OCR <-> risk model <-> AI advice)."""
from .production_bundle import (
    BUNDLE_MARKER,
    OCR_TO_LAB_ALIAS,
    ProductionRiskModel,
    is_production_bundle,
    load_production_model,
    run_bundle,
)

__all__ = [
    "BUNDLE_MARKER",
    "OCR_TO_LAB_ALIAS",
    "ProductionRiskModel",
    "is_production_bundle",
    "load_production_model",
    "run_bundle",
]
