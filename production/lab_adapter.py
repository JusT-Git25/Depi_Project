"""Universal lab-OCR -> CDC feature adapter.

Pipeline stage between the OCR JSON and `predict.predict_risk`:

    OCR JSON (any lab's naming) -> canonical labs -> CDC partial vector -> model

Handles:
  * arbitrary key names via alias map + fuzzy fallback (`difflib`)
  * three JSON shapes: {name: value}, {name: {value, unit}}, [{test, value, unit}]
  * unit conversion (mmol/L, lbs, cm/ft-in) and "120/80" BP strings
  * AHA cutoffs: HighBP=1 if sys>=130 or dia>=80; HighChol=1 if total>=200 mg/dL
  * missing fields -> NaN (never fabricated), with provenance per field

Glucose / HbA1c have NO slot in the current 21-feature CDC model, so they are
surfaced in `lab_signals` with a warning instead of being silently dropped.

For the full OCR -> prediction -> advice flow see `production.pipeline`.
"""
from __future__ import annotations

import difflib
import math
import re
from typing import Any

from production.clinical_rules import (
    evaluate_glycemic_signals,
    ifcc_to_ngsp,
    is_plausible,
    BOUNDS,
)

CDC_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# AHA-style cutoffs .
SYS_CUTOFF = 130.0
DIA_CUTOFF = 80.0
TOTAL_CHOL_CUTOFF = 200.0  # mg/dL
LDL_FALLBACK_CUTOFF = 130.0  # used only when total cholesterol is absent


def _norm_key(k: str) -> str:
    k = str(k).lower().strip()
    k = re.sub(r"[^a-z0-9]+", " ", k)
    return re.sub(r"\s+", " ", k).strip()


# canonical lab -> known normalized aliases across labs
ALIASES: dict[str, list[str]] = {
    "systolic": ["systolic", "sys", "sbp", "systolic bp", "systolic blood pressure",
                 "bp systolic", "blood pressure systolic", "sys bp"],
    "diastolic": ["diastolic", "dia", "dbp", "diastolic bp", "diastolic blood pressure",
                  "bp diastolic", "blood pressure diastolic", "dias bp"],
    "bp": ["bp", "blood pressure", "b p", "arterial pressure", "bp reading",
           "bloodpressure", "systolic diastolic"],
    "total_chol": ["total cholesterol", "total chol", "cholesterol total",
                   "cholesterol", "chol", "tchol", "tc", "serum cholesterol"],
    "hdl": ["hdl", "hdl cholesterol", "hdl c", "high density lipoprotein"],
    "ldl": ["ldl", "ldl cholesterol", "ldl c", "low density lipoprotein",
            "ldl calculated"],
    "triglycerides": ["triglycerides", "triglyceride", "trig", "tg"],
    "glucose_fasting": ["fasting glucose", "fbs", "fasting blood glucose", "fasting blood sugar",
                        "fasting blood sugar fbs", "fbg", "glucose fasting", "serum fasting glucose"],
    "glucose_random": ["random glucose", "rbs", "random blood glucose", "random blood sugar",
                       "casual glucose", "postprandial glucose", "ppbs", "2h pp", "glucose random"],
    "glucose": ["glucose", "blood glucose", "blood sugar", "serum glucose", "plasma glucose"],
    "hba1c": ["hba1c", "hb a1c", "a1c", "glycated hemoglobin", "glycosylated hemoglobin",
              "hemoglobin a1c"],
    "weight": ["weight", "wt", "body weight", "weight kg", "weight lbs"],
    "height": ["height", "ht", "stature", "height cm"],
    "bmi": ["bmi", "body mass index", "b m i"],
    "age": ["age", "patient age", "years old", "age years"],
    "sex": ["sex", "gender", "patient sex", "male female"],
    "smoker": ["smoker", "smoking", "tobacco", "cigarette", "smoke"],
    "heart": ["heart disease", "heart attack", "mi", "myocardial infarction",
              "coronary", "heartdiseaseorattack"],
    "stroke": ["stroke", "cva", "cerebrovascular"],
}
_ALIAS_LOOKUP = {_norm_key(a): canon for canon, names in ALIASES.items() for a in names}


def _canonical(name: str) -> tuple[str | None, bool]:
    """Map a raw lab name to canonical. Returns (canon, fuzzy_used)."""
    n = _norm_key(name)
    if n in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[n], False
    match = difflib.get_close_matches(n, list(_ALIAS_LOOKUP), n=1, cutoff=0.82)
    if match:
        return _ALIAS_LOOKUP[match[0]], True
    # last resort: substring hit (e.g. "LDL-C (calculated)")
    for alias, canon in _ALIAS_LOOKUP.items():
        if alias and (alias in n or n in alias):
            return canon, True
    return None, False


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_value(v: Any) -> tuple[float | None, str]:
    """Extract (number, unit-string) from OCR values like '200 mg/dL', 5.7, {'value':..}."""
    unit = ""
    if isinstance(v, dict):
        unit = str(v.get("unit", "") or "")
        v = v.get("value", v.get("result", v.get("val")))
    if isinstance(v, (int, float)):
        return (float(v) if not (isinstance(v, float) and math.isnan(v)) else None, unit)
    s = str(v).strip()
    if "/" in s and len(_NUM_RE.findall(s)) >= 2:  # leave "120/80" to BP parser
        return None, s
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return None, s
    return float(m.group()), (unit + " " + s).lower()


def _to_mg_dl(x: float, unit_blob: str, kind: str) -> float:
    u = unit_blob.lower()
    if "mmol" in u:
        return x * (38.67 if kind == "chol" else 18.015)
    return x


def _parse_bp_value(v: Any) -> tuple[float | None, float | None]:
    if isinstance(v, dict):
        v = v.get("value", v.get("result", v))
    s = str(v)
    nums = _NUM_RE.findall(s.replace(",", ""))
    if "/" in s and len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return None, None
    return None, None


def _parse_sex(v: Any) -> float | None:
    s = _norm_key(str(v) if not isinstance(v, dict) else str(v.get("value", "")))
    if s in {"1", "male", "m", "man", "boy", "masculino"}:
        return 1.0
    if s in {"0", "female", "f", "woman", "girl", "fem"}:
        return 0.0
    try:
        f = float(str(v))
        return 1.0 if f == 1 else (0.0 if f == 0 else None)
    except (ValueError, TypeError):
        return None


def _age_years_to_band(years: float) -> float:
    bounds = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, b in enumerate(bounds):
        if years < (bounds[i + 1] if i + 1 < len(bounds) else float("inf")):
            if years < 18:
                return 1.0
            return float(i + 1)
    return 13.0


def _iter_records(lab_json: Any) -> list[tuple[str, Any]]:
    """Flatten the three OCR shapes into [(raw_name, raw_value)]."""
    if isinstance(lab_json, dict):
        return list(lab_json.items())
    if isinstance(lab_json, list):
        out = []
        for r in lab_json:
            if isinstance(r, dict):
                name = r.get("test", r.get("test_name", r.get("name", r.get("analyte", ""))))
                out.append((str(name), r))
            else:
                out.append(("", r))
        return out
    raise ValueError("lab_json must be a dict or a list of {test, value, unit} records")


def normalize_labs(lab_json: Any) -> dict:
    """OCR JSON -> canonical labs + provenance. Never raises on unknown names."""
    canon: dict[str, float | str] = {}
    prov: dict[str, str] = {}
    unknown: list[str] = []
    fuzzy_used: list[str] = []
    for raw_name, raw_val in _iter_records(lab_json):
        c, fuzzy = _canonical(raw_name or "")
        if c is None:
            # maybe the VALUE holds "120/80"
            s, d = _parse_bp_value(raw_val)
            if s is not None:
                canon["systolic"], canon["diastolic"] = s, d
                prov["systolic"] = prov["diastolic"] = f"parsed '{raw_val}' as BP"
                continue
            unknown.append(str(raw_name))
            continue
        if fuzzy:
            fuzzy_used.append(str(raw_name))
        if c == "bp":
            s, d = _parse_bp_value(raw_val)
            if s is not None:
                canon["systolic"], canon["diastolic"] = s, d
                prov["systolic"] = prov["diastolic"] = f"from '{raw_name}'"
            continue
        if c == "sex":
            val = raw_val.get("value", raw_val) if isinstance(raw_val, dict) else raw_val
            parsed = _parse_sex(val)
            if parsed is not None:
                canon[c] = parsed
                prov[c] = f"from '{raw_name}'"
            continue
        num, ublob = _parse_value(raw_val)
        if num is None:
            unknown.append(str(raw_name))
            continue
        if c in {"total_chol", "hdl", "ldl", "triglycerides"}:
            num = _to_mg_dl(num, ublob, "chol")
        elif c in {"glucose", "glucose_fasting", "glucose_random"}:
            num = _to_mg_dl(num, ublob, "gluc")
        elif c == "hba1c":
            u_lower = ublob.lower()
            is_pct = "%" in u_lower or "pct" in u_lower or "percent" in u_lower
            if "mmol" in u_lower or (num > 20.0 and not is_pct):
                old_val = num
                num = round(ifcc_to_ngsp(num), 1)
                prov[c] = f"from '{raw_name}' (converted {old_val:g} mmol/mol IFCC -> {num}% NGSP)"
                canon[c] = num
                continue
        canon[c] = num
        prov[c] = f"from '{raw_name}'{' (fuzzy)' if fuzzy else ''}"
    # height/weight unit normalization: detect lbs / cm / ft-in from provenance blob
    return {"canonical": canon, "provenance": prov,
            "unknown_fields": unknown, "fuzzy_matched": fuzzy_used}


def _weight_to_kg(x: float) -> float:
    return x / 2.20462 if x > 150 else x  # >150 must be lbs, not kg


def _height_to_m(x: float) -> float:
    if x > 10:  # cm
        return x / 100.0
    if 2 < x < 9:  # feet like 5.9
        return x * 0.3048
    return x  # already meters


def labs_to_cdc(lab_json: Any) -> dict:
    """Full adapter: OCR JSON -> {cdc_partial, lab_signals, provenance, warnings}."""
    norm = normalize_labs(lab_json)
    c, prov = norm["canonical"], norm["provenance"]
    cdc: dict[str, float] = {f: float("nan") for f in CDC_FEATURES}
    warnings: list[str] = []

    if "systolic" in c or "diastolic" in c:
        s, d = c.get("systolic"), c.get("diastolic")
        flag = ((s is not None and s >= SYS_CUTOFF) or (d is not None and d >= DIA_CUTOFF))
        cdc["HighBP"] = 1.0 if flag else 0.0
        prov["HighBP"] = f"AHA rule sys>=130 or dia>=80 (sys={s}, dia={d})"
    if "total_chol" in c or "ldl" in c:
        if "total_chol" in c:
            cdc["HighChol"] = 1.0 if c["total_chol"] >= TOTAL_CHOL_CUTOFF else 0.0
            prov["HighChol"] = f"total {c['total_chol']:.0f} mg/dL vs 200"
        else:
            cdc["HighChol"] = 1.0 if c["ldl"] >= LDL_FALLBACK_CUTOFF else 0.0
            prov["HighChol"] = f"LDL fallback {c['ldl']:.0f} mg/dL vs 130"
        cdc["CholCheck"] = 1.0
        prov["CholCheck"] = "cholesterol panel present in labs"
    if "bmi" in c:
        cdc["BMI"] = float(c["bmi"])
        prov["BMI"] = prov.get("bmi", "direct BMI")
    elif "weight" in c and "height" in c:
        h = _height_to_m(float(c["height"]))
        w = _weight_to_kg(float(c["weight"]))
        if h > 0:
            cdc["BMI"] = w / (h * h)
            prov["BMI"] = f"computed {w:.1f}kg / {h:.2f}m^2"
    elif "weight" in c or "height" in c:
        warnings.append("Only one of weight/height present; BMI left missing.")
    for direct, cdc_name in [("smoker", "Smoker"), ("stroke", "Stroke"),
                             ("heart", "HeartDiseaseorAttack"), ("age", "Age"),
                             ("sex", "Sex")]:
        if direct in c:
            if direct == "age":
                cdc["Age"] = _age_years_to_band(float(c["age"]))
                prov["Age"] = f"{c['age']}y -> band {cdc['Age']:.0f}"
            else:
                cdc[cdc_name] = float(c[direct])
                prov[cdc_name] = prov.get(direct, "")
    # direct CDC passthrough (e.g. app survey already collected them).
    # Never clobber a value already derived from canonical labs (e.g. Age years->band).
    for raw_name, raw_val in _iter_records(lab_json):
        target = next((f for f in CDC_FEATURES if _norm_key(f) == _norm_key(raw_name)), None)
        if target is None or target in prov:
            continue
        try:
            v = raw_val.get("value", raw_val) if isinstance(raw_val, dict) else raw_val
            if target == "Age" and float(v) > 13:  # raw years, not a band
                cdc["Age"] = _age_years_to_band(float(v))
                prov["Age"] = f"{v}y -> band {cdc['Age']:.0f} (passthrough)"
            else:
                cdc[target] = float(v)  # type: ignore
                prov[target] = "direct CDC value"
        except (ValueError, TypeError, StopIteration):
            pass

    signals = {k: c[k] for k in ("glucose_fasting", "glucose_random", "glucose", "hba1c") if k in c}
    if signals:
        warnings.append(
            "Glucose/HbA1c extracted but NOT model inputs (CDC-21 has no glucose slot); "
            "evaluated via clinical ADA rules and shown in lab_signals/clinical_assessment.")

    clinical_assessment = evaluate_glycemic_signals(signals)
    if clinical_assessment and clinical_assessment.get("warnings"):
        warnings.extend(clinical_assessment["warnings"])

    filled = {k: v for k, v in cdc.items() if not math.isnan(v)}
    missing = [k for k, v in cdc.items() if math.isnan(v)]
    return {
        "cdc_partial": cdc,
        "filled": filled,
        "missing": missing,
        "lab_signals": signals,
        "clinical_assessment": clinical_assessment,
        "glycemic_alert": clinical_assessment,  # backward compatibility alias
        "provenance": prov,
        "unknown_fields": norm["unknown_fields"],
        "fuzzy_matched": norm["fuzzy_matched"],
        "warnings": warnings,
    }



def vectorize(cdc_partial: dict) -> list:
    """Ordered 21-vector (NaN where missing) ready for a NaN-tolerant model."""
    return [float(cdc_partial.get(f, float("nan"))) for f in CDC_FEATURES]
