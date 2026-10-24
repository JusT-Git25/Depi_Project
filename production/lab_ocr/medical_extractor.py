import json
import re
from pathlib import Path
from .schemas import LabResult, ReferenceRange

_MAPPING_PATH = Path(__file__).parent / "mappings" / "test_names.json"
try:
    TEST_NAME_MAP = {k.casefold(): v for k, v in json.loads(_MAPPING_PATH.read_text(encoding="utf-8")).items()}
except Exception:
    TEST_NAME_MAP = {}

QUALITATIVE = {"positive", "negative", "reactive", "non-reactive", "nonreactive", "trace", "detected", "not detected"}
FLAG_RE = re.compile(r"\b(HIGH|LOW|CRITICAL|H|L|HH|LL)\b", re.I)
RANGE_RE = re.compile(r"(?:ref(?:erence)?\s*(?:range)?\s*[:=]?\s*)?\(?\s*(-?\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:[.,]\d+)?)\s*\)?", re.I)
OPEN_RANGE_RE = re.compile(r"(?:ref(?:erence)?\s*(?:range)?\s*[:=]?\s*)?([<>]=?)\s*(-?\d+(?:[.,]\d+)?)", re.I)
RESULT_RE = re.compile(r"(?P<op><=|>=|<|>|=)?\s*(?P<value>-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)")
UNIT_RE = re.compile(r"(?i)(mg/dl|mmol/l|g/dl|g/l|u/l|iu/l|miu/l|ng/ml|pg/ml|fl|%|10\^?\d+\s*/?\s*(?:ul|µl|ml)|x10\^?\d+\s*/?\s*(?:ul|µl|ml)|cells/ul|mm/hr|seconds?)")

def _number(s: str) -> float:
    return float(s.replace(",", "."))

def _parse_range(text: str):
    m = RANGE_RE.search(text)
    if m:
        try:
            low, high = _number(m.group(1)), _number(m.group(2))
            if low <= high:
                return ReferenceRange(low=low, high=high, raw=m.group(0))
        except ValueError:
            pass
    m = OPEN_RANGE_RE.search(text)
    if m:
        try:
            return ReferenceRange(operator=m.group(1), low=_number(m.group(2)) if m.group(1).startswith(">") else None,
                                  high=_number(m.group(2)) if m.group(1).startswith("<") else None, raw=m.group(0))
        except ValueError:
            pass
    return None

def _parse_result(text: str):
    # Prefer a qualitative token when it appears as a standalone result.
    for word in QUALITATIVE:
        if re.search(rf"\b{re.escape(word)}\b", text, re.I):
            return word.title(), None
    m = RESULT_RE.search(text)
    if m:
        try:
            return _number(m.group("value")), m.group("op")
        except ValueError:
            return None, None
    return None, None

def extract_results(text: str, page_number: int, confidence: float | None = None) -> list[LabResult]:
    results = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" |\t")
        if not line or len(line) < 3:
            continue
        # Avoid headers and dates; require a colon, a known test label, or a likely table row.
        parts = re.split(r"\s*:\s*", line, maxsplit=1)
        if len(parts) == 2:
            original_name, tail = parts[0].strip(), parts[1].strip()
        else:
            # A conservative table heuristic: label followed by a numeric/qualitative token.
            m = re.match(r"^([A-Za-z][A-Za-z0-9 /()_.%-]{1,45}?)\s{1,}(.+)$", line)
            if not m:
                continue
            original_name, tail = m.group(1).strip(), m.group(2).strip()
        normalized = TEST_NAME_MAP.get(original_name.casefold())
        # Keep plausible test rows only. Unknown names are retained if a result can be parsed.
        value, operator = _parse_result(tail)
        if value is None:
            continue
        # If a range is present, don't confuse its lower bound for the result by stripping it first.
        range_obj = _parse_range(tail)
        unit_match = UNIT_RE.search(tail)
        unit = unit_match.group(0) if unit_match else None
        flag_match = FLAG_RE.search(tail)
        flag = flag_match.group(1).upper() if flag_match else None
        warnings = []
        status = "needs_review"
        if confidence is not None and confidence < 55:
            warnings.append("Low OCR confidence.")
        if not normalized:
            warnings.append("Test name is not mapped; verify the original label.")
        if unit is None and isinstance(value, (float, int)):
            warnings.append("Unit not detected; verify against the source report.")
        if range_obj is None and re.search(r"ref|range", tail, re.I):
            warnings.append("Reference range appears ambiguous.")
        if len(re.findall(r"-?\d+(?:[.,]\d+)?", tail)) > 2:
            warnings.append("Multiple numeric candidates detected; verify value and range.")
        # Only mark as validated when source is unambiguous enough for a preliminary technical status.
        if normalized and not warnings and confidence is not None and confidence >= 80:
            status = "validated"
        results.append(LabResult(
            test_name=normalized or original_name,
            original_test_name=original_name,
            raw_result_text=line,
            value=value,
            value_operator=operator,
            unit=unit,
            reference_range=range_obj,
            abnormal_flag=flag,
            page_number=page_number,
            confidence=confidence,
            validation_status=status,
            validation_warnings=warnings
        ))
    return results
