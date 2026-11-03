# Model Calibration - Documentation

## Objective
Improve the reliability of the predicted probabilities from the LightGBM model, ensuring they accurately reflect the actual risk of diabetes.

## Methods Tested
1. **Baseline** — No calibration (raw LightGBM probabilities).
2. **Isotonic Regression** — Non-parametric calibration.
3. **Sigmoid (Platt Scaling)** — Parametric calibration.

## Results

| Model | ROC-AUC | PR-AUC | Brier Score |
|-------|---------|--------|-------------|
| Baseline | 0.8269 | 0.4242 | 0.1745 |
| Isotonic | 0.8271 | 0.4227 | 0.0974 |
| **Sigmoid** | **0.8272** | **0.4234** | **0.0973** |

## Final Choice
**Sigmoid (Platt Scaling)** was selected as the final calibration method.

## Rationale
- **Brier Score improved from 0.1745 to 0.0973** (~44% reduction), meaning probabilities are now much closer to actual outcomes.
- **PR-AUC slightly higher** than Isotonic (0.4234 vs 0.4227).
- **Faster training** compared to Isotonic.
- **Less prone to overfitting**, especially on smaller datasets.

## Files
- `calibration.py` — Full calibration script (Baseline, Isotonic, Sigmoid).
- `calibration_curve.png` — Calibration curves for all three models.

## Conclusion
Calibration significantly improved the reliability of the model's probabilities, making it more suitable for clinical risk awareness applications.