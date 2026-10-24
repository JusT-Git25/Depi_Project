"""Clinical laboratory evaluation rules for diabetes screening.

Single source of truth for:
  - Physiological plausibility bounds (guarding against OCR misreads)
  - Fasting vs random/casual glucose thresholds (ADA guidelines)
  - HbA1c unit conversion (IFCC mmol/mol -> NGSP %) and thresholds
  - Non-diagnostic clinical safety language (avoiding false diagnosis claims)
"""
from __future__ import annotations

import math
from typing import Any, Mapping

# Physiological plausibility bounds: values outside these are likely OCR artifacts.
BOUNDS = {
    "hba1c": (3.0, 20.0),            # NGSP %
    "glucose": (20.0, 600.0),         # mg/dL
    "systolic": (50.0, 260.0),        # mmHg
    "diastolic": (30.0, 150.0),       # mmHg
    "total_chol": (50.0, 800.0),      # mg/dL
    "bmi": (10.0, 90.0),              # kg/m^2
}

# ADA Thresholds
FASTING_GLUCOSE_PREDIABETES = 100.0
FASTING_GLUCOSE_ELEVATED = 126.0

RANDOM_GLUCOSE_PREDIABETES = 140.0
RANDOM_GLUCOSE_ELEVATED = 200.0

HBA1C_PREDIABETES = 5.7
HBA1C_ELEVATED = 6.5


def ifcc_to_ngsp(mmol_mol: float) -> float:
    """Convert IFCC HbA1c (mmol/mol) to NGSP (%)."""
    return (mmol_mol * 0.09148) + 2.152


def is_plausible(test: str, value: float) -> bool:
    """True if value is within known physiological bounds."""
    if test not in BOUNDS:
        return True
    lo, hi = BOUNDS[test]
    return lo <= value <= hi


def evaluate_glycemic_signals(signals: Mapping[str, Any] | None) -> dict | None:
    """Evaluate laboratory glycemic signals against ADA criteria.

    Returns None if no glycemic markers are present.
    Returns structured assessment dict:
      - status: 'elevated_range' | 'prediabetic_range' | 'normal' | 'needs_recheck'
      - flagged: bool (True if prediabetic or elevated)
      - clinical_level: 'High' | 'Elevated' | 'Low' | None
      - summary: English explanation (safe, non-diagnostic wording)
      - summary_ar: Arabic explanation
      - recommendation: Clinical guidance in English
      - recommendation_ar: Clinical guidance in Arabic
      - details: list of analyzed tests with values, units, and interpretations
      - warnings: list of clinical or plausibility warnings
    """
    if not signals:
        return None

    details: list[dict] = []
    warnings: list[str] = []
    has_implausible = False
    highest_severity = 0  # 0: normal, 1: prediabetes, 2: elevated/above threshold

    # 1. Evaluate HbA1c
    if "hba1c" in signals:
        try:
            val = float(signals["hba1c"])
            if not is_plausible("hba1c", val):
                has_implausible = True
                warnings.append(
                    f"HbA1c value {val:g}% is outside physiological bounds "
                    f"[{BOUNDS['hba1c'][0]}-{BOUNDS['hba1c'][1]}%]; likely an OCR error. Please verify."
                )
                details.append({
                    "test": "HbA1c",
                    "value": f"{val:g}%",
                    "numeric_value": val,
                    "severity": -1,
                    "interpretation": "Value outside plausible physiological range — verify report",
                    "interpretation_ar": "القيمة خارج النطاق الفسيولوجي المعقول — يُرجى مراجعة التقرير",
                })
            elif val >= HBA1C_ELEVATED:
                highest_severity = max(highest_severity, 2)
                details.append({
                    "test": "HbA1c",
                    "value": f"{val:.1f}%",
                    "numeric_value": val,
                    "severity": 2,
                    "interpretation": "Above reference threshold (HbA1c >= 6.5%) — requires clinical confirmation",
                    "interpretation_ar": "أعلى من الحد المرجعي (HbA1c >= 6.5%) — يتطلب تأكيدًا طبيًا",
                })
            elif val >= HBA1C_PREDIABETES:
                highest_severity = max(highest_severity, 1)
                details.append({
                    "test": "HbA1c",
                    "value": f"{val:.1f}%",
                    "numeric_value": val,
                    "severity": 1,
                    "interpretation": "Prediabetes reference range (HbA1c 5.7–6.4%)",
                    "interpretation_ar": "في نطاق مرحلة ما قبل السكري (HbA1c 5.7–6.4%)",
                })
            else:
                details.append({
                    "test": "HbA1c",
                    "value": f"{val:.1f}%",
                    "numeric_value": val,
                    "severity": 0,
                    "interpretation": "Normal reference range (HbA1c < 5.7%)",
                    "interpretation_ar": "في النطاق المرجعي الطبيعي (HbA1c < 5.7%)",
                })
        except (ValueError, TypeError):
            pass

    # 2. Evaluate Glucose (Fasting vs Random / Unspecified)
    glucose_entries = [
        ("glucose_fasting", True),
        ("glucose_random", False),
        ("glucose", False),  # unspecified defaults to conservative random criteria
    ]
    seen_glucose = False
    for g_key, is_fasting in glucose_entries:
        if g_key in signals and not seen_glucose:
            seen_glucose = True
            try:
                val = float(signals[g_key])
                label = "Fasting Blood Glucose" if is_fasting else (
                    "Random Blood Glucose" if g_key == "glucose_random" else "Blood Glucose"
                )
                label_ar = "سكر الدم الصائم" if is_fasting else (
                    "سكر الدم العشوائي" if g_key == "glucose_random" else "سكر الدم"
                )

                if not is_plausible("glucose", val):
                    has_implausible = True
                    warnings.append(
                        f"{label} value {val:g} mg/dL is outside physiological bounds "
                        f"[{BOUNDS['glucose'][0]}-{BOUNDS['glucose'][1]} mg/dL]; verify report."
                    )
                    details.append({
                        "test": label,
                        "value": f"{val:g} mg/dL",
                        "numeric_value": val,
                        "severity": -1,
                        "interpretation": "Value outside plausible physiological range — verify report",
                        "interpretation_ar": "القيمة خارج النطاق الفسيولوجي المعقول — يُرجى مراجعة التقرير",
                    })
                else:
                    elevated_cutoff = FASTING_GLUCOSE_ELEVATED if is_fasting else RANDOM_GLUCOSE_ELEVATED
                    prediabetes_cutoff = FASTING_GLUCOSE_PREDIABETES if is_fasting else RANDOM_GLUCOSE_PREDIABETES

                    if val >= elevated_cutoff:
                        highest_severity = max(highest_severity, 2)
                        details.append({
                            "test": label,
                            "value": f"{val:.0f} mg/dL",
                            "numeric_value": val,
                            "severity": 2,
                            "interpretation": f"Above reference threshold ({label} >= {elevated_cutoff:.0f} mg/dL) — requires confirmation",
                            "interpretation_ar": f"أعلى من الحد المرجعي ({label_ar} >= {elevated_cutoff:.0f} ملغم/ديسيلتر) — يتطلب تأكيدًا طبيًا",
                        })
                    elif val >= prediabetes_cutoff:
                        highest_severity = max(highest_severity, 1)
                        details.append({
                            "test": label,
                            "value": f"{val:.0f} mg/dL",
                            "numeric_value": val,
                            "severity": 1,
                            "interpretation": f"Impaired/prediabetes range ({label} {prediabetes_cutoff:.0f}–{elevated_cutoff - 1:.0f} mg/dL)",
                            "interpretation_ar": f"في نطاق ما قبل السكري ({label_ar} {prediabetes_cutoff:.0f}–{elevated_cutoff - 1:.0f} ملغم/ديسيلتر)",
                        })
                    else:
                        details.append({
                            "test": label,
                            "value": f"{val:.0f} mg/dL",
                            "numeric_value": val,
                            "severity": 0,
                            "interpretation": f"Normal reference range ({label} < {prediabetes_cutoff:.0f} mg/dL)",
                            "interpretation_ar": f"في النطاق الطبيعي ({label_ar} < {prediabetes_cutoff:.0f} ملغم/ديسيلتر)",
                        })
            except (ValueError, TypeError):
                pass

    if not details:
        return None

    if has_implausible:
        return {
            "status": "needs_recheck",
            "flagged": False,
            "clinical_level": None,
            "summary": "One or more lab values fall outside plausible physiological limits (possible OCR artifact). Please recheck the report.",
            "summary_ar": "قيمة أو أكثر من نتائج التحاليل خارج الحدود الفسيولوجية المعقولة (احتمال خطأ في قراءة التقرير). يُرجى التحقق من القيم.",
            "recommendation": "Review the uploaded lab document and verify the highlighted values with a healthcare provider.",
            "recommendation_ar": "راجع تقرير الفحص المرفوع وتأكد من القيم الموضحة مع الطبيب أو أخصائي التحاليل.",
            "details": details,
            "warnings": warnings,
        }

    if highest_severity == 2:
        return {
            "status": "elevated_range",
            "flagged": True,
            "clinical_level": "High",
            "summary": "Laboratory glycemic findings are above the standard reference threshold (requires clinical confirmation).",
            "summary_ar": "نتائج تحاليل السكر المخبرية أعلى من الحدود المرجعية القياسية (يتطلب تأكيدًا وتقييمًا سريريًا).",
            "recommendation": "Prompt consultation with a physician for clinical evaluation and confirmatory testing is advised.",
            "recommendation_ar": "يُنصح بمراجعة الطبيب المختص لعمل فحص تأكيدي وتقييم سريري شامل.",
            "details": details,
            "warnings": warnings,
        }
    elif highest_severity == 1:
        return {
            "status": "prediabetic_range",
            "flagged": True,
            "clinical_level": "Elevated",
            "summary": "Laboratory glycemic findings indicate prediabetes range (impaired glucose regulation).",
            "summary_ar": "تشير نتائج تحاليل السكر إلى نطاق مرحلة ما قبل السكري (اضطراب في تنظيم السكر).",
            "recommendation": "Lifestyle modifications and regular clinical follow-up with a doctor are recommended to prevent progression.",
            "recommendation_ar": "يُنصح باتباع نمط حياة صحي ومتابعة طبية منتظمة مع الطبيب لتفادي تطور الحالة.",
            "details": details,
            "warnings": warnings,
        }
    else:
        return {
            "status": "normal",
            "flagged": False,
            "clinical_level": "Low",
            "summary": "Laboratory glycemic markers are within the normal reference range.",
            "summary_ar": "نتائج تحاليل السكر تقع ضمن النطاق المرجعي الطبيعي.",
            "recommendation": "Continue healthy lifestyle habits and regular preventative health check-ups.",
            "recommendation_ar": "حافظ على العادات الصحية واستمر في إجراء الفحوصات الدورية الوقائية.",
            "details": details,
            "warnings": warnings,
        }
