"""Shared data helpers for the COPD scripts.

The cleaning itself (missing values, outliers, correlated column) is done and explained in
COPD_EDA_Cleaning.ipynb, which writes COPD_clean.csv and artifacts/copd_cleaning.json.
LazyPredict, Grid Search and the final model all read that file through these helpers,
so every step uses exactly the same split and the same balancing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CLEAN_CSV = ROOT / "COPD_clean.csv"
CLEANING_JSON = ROOT / "artifacts" / "copd_cleaning.json"


def cleaning_settings() -> dict:
    if not CLEANING_JSON.exists():
        raise FileNotFoundError(
            f"{CLEANING_JSON} not found. Run COPD_EDA_Cleaning.ipynb first (it creates the clean data).")
    return json.loads(CLEANING_JSON.read_text())


def load_clean():
    """Return (X, y) from the cleaned data produced by the EDA notebook."""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"{CLEAN_CSV} not found. Run COPD_EDA_Cleaning.ipynb first.")
    s = cleaning_settings()
    df = pd.read_csv(CLEAN_CSV)
    return df[s["features"]], df[s["target"]]


def get_train_test(test_size=0.20, random_state=42):
    """Same 80/20 stratified split as the notebook and every other COPD script."""
    X, y = load_clean()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def balance(X, y, random_state=42):
    """Balance 50/50 by random under-sampling. Use on TRAINING data only, never on test."""
    Xb, yb = RandomUnderSampler(random_state=random_state).fit_resample(X, y)
    return Xb, yb


def prepare_patient(patient: dict) -> pd.DataFrame:
    """Raw survey answers for one patient -> one-row frame ready for the model.

    Accepts the original survey fields (including SMOKE100, which is no longer a model feature
    but is used to fill SMOKDAY2 for never-smokers), applies the same clipping and filling
    as the notebook, and returns the model columns in the right order.
    """
    s = cleaning_settings()
    row = {k: (np.nan if v is None else v) for k, v in patient.items()}
    if row.get("SMOKE100") == 2:                      # never smoked -> not asked about current smoking
        row["SMOKDAY2"] = 0
    df = pd.DataFrame([row]).reindex(columns=s["features"]).apply(pd.to_numeric, errors="raise")
    lo, hi = s["bmi_clip"]
    df["BMI"] = df["BMI"].clip(lo, hi)
    missing = [c for c in s["features"] if pd.isna(df.at[0, c])]
    return df.fillna(s["fill_values"]), missing
