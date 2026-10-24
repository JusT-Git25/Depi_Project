"""Create a DEMO model so you can try the whole flow end to end.

The data is synthetic and the labels come from a made-up rule. The output has NO medical meaning.
Replace it with your real trained model and a spec that matches it (see models/model_spec.example.json).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)

rng = np.random.default_rng(0)
n = 400
X = pd.DataFrame({"glucose": rng.normal(100, 25, n), "hba1c": rng.normal(5.6, 0.8, n)})
y = ((X["glucose"] > 110) & (X["hba1c"] > 5.7)).astype(int)  # made-up rule
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
joblib.dump(model, out / "demo_model.joblib")

spec = {
    "name": "DEMO model (synthetic data)", "version": "demo", "demo": True,
    "model_path": "demo_model.joblib",
    "feature_order": ["glucose", "hba1c"],
    "feature_mapping": {"Glucose": "glucose", "HbA1c": "hba1c"},
    "expected_units": {"glucose": "mg/dL", "hba1c": "%"},
    "labels": {"0": "Lower risk (demo)", "1": "Higher risk (demo)"},
}
(out / "demo_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
print("Wrote", out / "demo_model.joblib", "and", out / "demo_spec.json")
