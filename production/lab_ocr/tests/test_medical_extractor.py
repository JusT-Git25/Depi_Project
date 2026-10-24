from lab_ocr.medical_extractor import extract_results

def test_extract_common_result():
    results = extract_results("Hemoglobin: 12.5 g/dL (Reference Range: 12.0-16.0)", 1, 92)
    assert results
    assert results[0].original_test_name == "Hemoglobin"
    assert results[0].value == 12.5
    assert results[0].unit.lower() == "g/dl"
    assert results[0].reference_range.low == 12.0
    assert results[0].reference_range.high == 16.0

def test_qualitative_result():
    results = extract_results("Urine nitrite: Negative", 2, 90)
    assert results and results[0].value == "Negative"

def test_does_not_infer_abnormality():
    results = extract_results("Glucose: 999 mg/dL", 1, 95)
    assert results
    assert results[0].abnormal_flag is None


def test_common_ocr_misread_of_hba1c_is_mapped():
    # Tesseract often reads the digit 1 as a lowercase L.
    results = extract_results("HbAlc: 7.0 % (Reference Range: 4.0-5.6)", 1, 90)
    assert results[0].test_name == "HbA1c"
