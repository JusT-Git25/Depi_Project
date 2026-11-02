"""Grid Search for the best LightGBM parameters (model chosen with COPD_LazyPredict.ipynb).

The training data is balanced (under-sampling) INSIDE each cross-validation fold, so the
validation folds keep the natural 8% COPD rate and nothing leaks from the balancing step.
Scoring is PR-AUC (average_precision), the metric that matters most for a rare disease.

Run from the repo root (after COPD_EDA_Cleaning.ipynb created COPD_clean.csv):
    python COPD_GridSearch.py
"""
import json
import time

import lightgbm as lgb
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from copd_data import ROOT, get_train_test

X_train, X_test, y_train, y_test = get_train_test()
print("Train:", X_train.shape, "| Test:", X_test.shape)

pipe = Pipeline([
    ("balance", RandomUnderSampler(random_state=42)),
    ("lgbm", lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=2)),
])

param_grid = {
    "lgbm__n_estimators": [200, 400],
    "lgbm__learning_rate": [0.03, 0.06],
    "lgbm__num_leaves": [7, 15, 31],
    "lgbm__min_child_samples": [50, 200],
    "lgbm__colsample_bytree": [0.8, 1.0],
}
n_fits = 2 * 2 * 3 * 2 * 2 * 3
print(f"\nGrid Search: {n_fits // 3} combinations x 3 folds = {n_fits} fits ...")

search = GridSearchCV(
    pipe, param_grid, scoring="average_precision",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=1, verbose=1,
)
t0 = time.time()
search.fit(X_train, y_train)
print(f"Done in {time.time() - t0:.0f}s")

best = {k.replace("lgbm__", ""): v for k, v in search.best_params_.items()}
print("\nBest Parameters:")
print(best)
print("\nBest Cross-Validation PR-AUC:", round(search.best_score_, 4))

p = search.best_estimator_.predict_proba(X_test)[:, 1]
test = {"ROC-AUC": roc_auc_score(y_test, p), "PR-AUC": average_precision_score(y_test, p)}
print("\nTest Set Results (untouched, natural 8% COPD):")
print(test)

out = ROOT / "artifacts" / "copd_best_params.json"
out.write_text(json.dumps({"best_params": best, "cv_pr_auc": search.best_score_, "test": test}, indent=2))
print(f"\nSaved {out.relative_to(ROOT)}")
