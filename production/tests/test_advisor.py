"""Tests for advisor.py. They use a fake AI client, so no API key or internet is needed.
Run:  python -m pytest tests -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `production`
import production.advisor as advisor  # noqa: E402

PATIENT = dict(HighBP=1, HighChol=1, CholCheck=1, BMI=38, Smoker=1, Stroke=0, HeartDiseaseorAttack=1,
               PhysActivity=0, Fruits=0, Veggies=0, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=1,
               GenHlth=5, MentHlth=10, PhysHlth=25, DiffWalk=1, Sex=1, Age=11, Education=3, Income=2)
PREDICTION = {
    "risk_score": 0.7058, "risk_percent": "70.6%", "risk_level": "High", "flagged_for_screening": True,
    "screening_threshold": 0.1348, "recommendation": "Arrange a confirmatory blood test.",
    "top_risk_factors": [{"feature": "BMI", "value": 38.0, "contribution": 0.769},
                         {"feature": "GenHlth", "value": 5.0, "contribution": 0.765},
                         {"feature": "HighBP", "value": 1.0, "contribution": 0.635}],
}
GOOD_REPLY = {
    "summary": "Your estimated risk is high.",
    "key_factors": [{"factor": "BMI", "explanation": "A higher BMI raises risk."}],
    "lifestyle_tips": ["Walk daily.", "Eat more vegetables."],
    "next_steps": ["Get an HbA1c test."],
    "when_to_seek_care": "If you feel very thirsty.",
}


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create and records the request."""

    def __init__(self, reply=None, error=None):
        self.calls, self._reply, self._error = [], reply, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply)])


@pytest.fixture(autouse=True)
def isolate_environment_keys(monkeypatch):
    """Ensure shell environment keys never leak into unit tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)



def test_payload_is_minimal_and_readable():
    payload = advisor.build_payload(PREDICTION, PATIENT)
    text = json.dumps(payload)
    assert payload["estimated_diabetes_risk"] == "70.6%"
    assert payload["top_risk_factors"][0] == {"factor": "Body mass index (BMI)", "patient_value": "38", "impact": "strong"}
    assert payload["patient_profile"]["Age"] == "70-74"
    assert payload["patient_profile"]["Sex"] == "male"
    for unneeded in ("Income", "Education", "AnyHealthcare", "NoDocbcCost"):
        assert advisor._label(unneeded) not in payload["patient_profile"], unneeded
    assert "Income" not in text


def test_ai_success_path():
    client = FakeClient(json.dumps(GOOD_REPLY))
    out = advisor.generate_advice(PREDICTION, PATIENT, "en", model="test-model", client=client)
    assert out["source"] == "ai" and out["model"] == "test-model"
    assert out["summary"] == GOOD_REPLY["summary"] and out["lifestyle_tips"] == GOOD_REPLY["lifestyle_tips"]
    assert out["clinical_recommendation"] == PREDICTION["recommendation"]
    assert out["disclaimer"] == advisor.DISCLAIMER["en"]
    sent = client.calls[0]
    assert sent["model"] == "test-model" and sent["system"] == advisor.SYSTEM_PROMPT
    prompt = sent["messages"][0]["content"]
    assert "70.6%" in prompt and "Write the advice in English" in prompt


def test_arabic_language_is_requested():
    client = FakeClient(json.dumps(GOOD_REPLY))
    out = advisor.generate_advice(PREDICTION, PATIENT, "ar", client=client)
    assert "Write the advice in Arabic" in client.calls[0]["messages"][0]["content"]
    assert out["disclaimer"] == advisor.DISCLAIMER["ar"]


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "Here is the advice:\n{}\nHope it helps!", "{}"])
def test_parse_tolerates_fences_and_chatter(wrap):
    parsed = advisor.parse_advice(wrap.format(json.dumps(GOOD_REPLY)) if "{}" in wrap else wrap)
    assert parsed["summary"] == GOOD_REPLY["summary"]


@pytest.mark.parametrize("bad", ["not json at all", "{}", '{"summary": "x"}', '{"summary": "x", "lifestyle_tips": "str"}'])
def test_malformed_reply_falls_back(bad):
    out = advisor.generate_advice(PREDICTION, PATIENT, "en", client=FakeClient(bad))
    assert out["source"] == "fallback" and out["lifestyle_tips"] and out["fallback_reason"]


def test_api_error_falls_back_without_raising():
    out = advisor.generate_advice(PREDICTION, PATIENT, "en", client=FakeClient(error=TimeoutError("boom")))
    assert out["source"] == "fallback" and "TimeoutError" in out["fallback_reason"]
    assert out["disclaimer"] == advisor.DISCLAIMER["en"]


def test_missing_key_falls_back(monkeypatch):
    monkeypatch.setattr(advisor, "load_api_key", lambda: None)
    out = advisor.generate_advice(PREDICTION, PATIENT, "ar")
    assert out["source"] == "fallback" and advisor.API_KEY_ENV in out["fallback_reason"]
    assert out["language"] == "ar" and "مرتفعة" in out["summary"]


def test_key_is_scrubbed_from_errors(monkeypatch):
    monkeypatch.setattr(advisor, "load_api_key", lambda: "sk-secret-123")
    monkeypatch.setattr(advisor, "_call_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad key sk-secret-123")))
    out = advisor.generate_advice(PREDICTION, PATIENT, "en")
    assert "sk-secret-123" not in json.dumps(out) and "***" in out["fallback_reason"]


def test_load_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv(advisor.API_KEY_ENV, "  sk-abc  ")
    assert advisor.load_api_key() == "sk-abc"


def test_fallback_tips_follow_the_profile():
    healthy = dict(PATIENT, BMI=22, Smoker=0, PhysActivity=1, HighBP=0, HighChol=0, Fruits=1, Veggies=1)
    low = {**PREDICTION, "risk_level": "Low", "risk_percent": "2.0%", "flagged_for_screening": False,
           "top_risk_factors": []}
    out = advisor.fallback_advice(low, healthy, "en")
    assert "weight loss" not in " ".join(out["lifestyle_tips"]) and "smoking" not in " ".join(out["lifestyle_tips"])
    assert out["lifestyle_tips"], "always at least one generic tip"


def test_input_validation():
    with pytest.raises(ValueError):
        advisor.generate_advice(PREDICTION, PATIENT, "fr")
    with pytest.raises(ValueError):
        advisor.generate_advice({"risk_level": "High"}, PATIENT, "en")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "artifacts" / "diabetes_risk_model.joblib").exists(),
                    reason="model not trained")
def test_end_to_end_with_real_model_single_and_batch():
    client = FakeClient(json.dumps(GOOD_REPLY))
    one = advisor.predict_with_advice(PATIENT, client=client)
    assert one["risk_level"] == "High" and one["advice"]["source"] == "ai"
    batch = advisor.predict_with_advice([PATIENT, dict(PATIENT, BMI=22, HighBP=0, Age=3, GenHlth=1)], client=client)
    assert len(batch) == 2 and all("advice" in r for r in batch)
    assert batch[0]["risk_score"] > batch[1]["risk_score"]


def test_advisor_payload_and_fallback_includes_lab_signals():
    pred_with_labs = {
        **PREDICTION,
        "lab_signals": {"hba1c": 7.4, "glucose": 145.0},
        "clinical_assessment": {
            "status": "elevated_range",
            "flagged": True,
            "summary": "Laboratory findings are above the reference threshold.",
            "summary_ar": "نتائج تحاليل السكر المخبرية أعلى من الحدود المرجعية.",
            "recommendation": "Prompt consultation with a physician is advised.",
            "recommendation_ar": "يُنصح بمراجعة الطبيب المختص.",
            "details": [
                {
                    "test": "HbA1c",
                    "value": "7.4%",
                    "severity": 2,
                    "interpretation": "Above reference threshold (HbA1c >= 6.5%) — requires clinical confirmation",
                    "interpretation_ar": "أعلى من الحد المرجعي (HbA1c >= 6.5%) — يتطلب تأكيدًا طبيًا",
                }
            ],
        }
    }
    payload = advisor.build_payload(pred_with_labs, PATIENT)
    assert "laboratory_findings" in payload
    assert payload["laboratory_findings"]["HbA1c"] == "7.4%"
    assert payload["laboratory_findings"]["Blood Glucose"] == "145 mg/dL"
    assert "clinical_assessment" in payload

    # Test fallback advice incorporates them
    fb_en = advisor.fallback_advice(pred_with_labs, PATIENT, "en")
    assert any("HbA1c" in f["factor"] for f in fb_en["key_factors"])
    assert any("doctor" in s.lower() for s in fb_en["next_steps"])

    fb_ar = advisor.fallback_advice(pred_with_labs, PATIENT, "ar")
    assert any("HbA1c" in f["factor"] or "التراكمي" in f["factor"] for f in fb_ar["key_factors"])
    assert any("الطبيب" in s for s in fb_ar["next_steps"])


def test_provider_detection(monkeypatch):
    # Gemini detection
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-123")
    prov, key, err = advisor.get_provider_and_key()
    assert prov == "gemini" and key == "gemini-key-123"
    assert advisor.load_api_key() == "gemini-key-123"

    # OpenAI detection
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-456")
    prov, key, err = advisor.get_provider_and_key()
    assert prov == "openai" and key == "openai-key-456"
    assert advisor.load_api_key() == "openai-key-456"

    # Explicit AI_PROVIDER
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key-789")
    prov, key, err = advisor.get_provider_and_key()
    assert prov == "gemini" and key == "goog-key-789"


def test_explicit_provider_missing_key_does_not_borrow_other_keys(monkeypatch):
    # Only Anthropic key is set, but user configured AI_PROVIDER=gemini
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    prov, key, err = advisor.get_provider_and_key()
    assert prov == "gemini"
    assert key is None
    assert "GEMINI_API_KEY" in err

    out = advisor.generate_advice(PREDICTION, PATIENT, "en")
    assert out["source"] == "fallback"
    assert "GEMINI_API_KEY" in out["fallback_reason"]


def test_gemini_call_uses_header_not_query_param(monkeypatch):
    recorded_requests = []

    class FakeResponse:
        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": json.dumps(GOOD_REPLY)}]}}]
            }).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        recorded_requests.append(req)
        return FakeResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    res = advisor._call_llm("sys", "user", api_key="secret-gemini-key", model="gemini-2.0-flash", provider="gemini")
    assert len(recorded_requests) == 1
    req = recorded_requests[0]
    # Verify key is NOT in query string!
    assert "secret-gemini-key" not in req.full_url
    # Verify key IS in x-goog-api-key header!
    assert req.headers.get("X-goog-api-key") == "secret-gemini-key"


