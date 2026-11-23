"""Scikit-learn compatible clinical feature engineering transformer for ILPD.

All derived features are row-wise physiological calculations based on standard
hepatology interpretations of liver function tests (LFTs):
- De Ritis Ratio (AST/ALT = Sgot / Sgpt)
- Direct-to-Total Bilirubin Ratio (DB / TB)
- Indirect (Unconjugated) Bilirubin (max(TB - DB, 0))
- Estimated Globulin (max(TP - ALB, 0.1))
- Log1p transforms for strongly right-skewed serum enzymes and bilirubin
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from liver_disease.config import RAW_FEATURE_COLUMNS

SKEWED_ENZYME_COLUMNS: List[str] = ["TB", "DB", "Alkphos", "Sgpt", "Sgot"]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-free row-wise feature transformer for ILPD predictors.

    Parameters
    ----------
    add_clinical_ratios : bool, default=True
        Whether to append standard LFT derived biomarkers (De Ritis ratio,
        DB/TB ratio, indirect bilirubin, estimated globulin).
    log_transform_skewed : bool, default=True
        Whether to apply ``np.log1p`` in-place to right-skewed serum biomarkers
        (TB, DB, Alkphos, Sgpt, Sgot) after computing physiological ratios.
    """

    def __init__(
        self,
        add_clinical_ratios: bool = True,
        log_transform_skewed: bool = True,
    ) -> None:
        self.add_clinical_ratios = add_clinical_ratios
        self.log_transform_skewed = log_transform_skewed

    def fit(self, X: pd.DataFrame, y: object = None) -> "ClinicalFeatureEngineer":
        """Stateless fit; validates presence of expected ILPD feature columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature engineering: {missing}")
        self.feature_names_in_ = np.array(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return transformed DataFrame without mutating input."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")

        df = X[RAW_FEATURE_COLUMNS].copy()

        # Ensure numeric dtypes before ratios
        for col in RAW_FEATURE_COLUMNS:
            if col != "Gender":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.add_clinical_ratios:
            eps = 1e-6
            # De Ritis ratio (AST / ALT) - log1p stabilized for extreme ratios
            df["AST_ALT_Ratio"] = np.log1p(
                np.clip(df["Sgot"] / np.maximum(df["Sgpt"], eps), 0.0, 50.0)
            )
            # Conjugated fraction of total bilirubin
            df["DB_TB_Ratio"] = np.clip(df["DB"] / np.maximum(df["TB"], eps), 0.0, 1.5)
            # Unconjugated (indirect) bilirubin
            df["Indirect_Bilirubin"] = np.log1p(np.maximum(df["TB"] - df["DB"], 0.0))
            # Estimated serum globulin (TP - ALB)
            df["Globulin"] = np.maximum(df["TP"] - df["ALB"], 0.1)

        if self.log_transform_skewed:
            for col in SKEWED_ENZYME_COLUMNS:
                df[col] = np.log1p(np.maximum(df[col], 0.0))

        return df

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names in deterministic order."""
        cols = list(RAW_FEATURE_COLUMNS)
        if self.add_clinical_ratios:
            cols.extend(["AST_ALT_Ratio", "DB_TB_Ratio", "Indirect_Bilirubin", "Globulin"])
        return np.array(cols, dtype=object)
