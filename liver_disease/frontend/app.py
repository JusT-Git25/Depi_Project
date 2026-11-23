"""Streamlit demonstration prototype for the Liver Disease Prediction System.

Architecture rule:
This frontend communicates EXCLUSIVELY with the FastAPI REST backend over HTTP.
It intentionally contains zero ML model loading or inference logic so that the
FastAPI service remains independently deployable and reusable by external backends.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Liver Disease Prediction Prototype",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_API_KEY = os.getenv("LIVER_API_KEY", "")

MEDICAL_DISCLAIMER = (
    "**ACADEMIC RESEARCH PROTOTYPE — NOT FOR CLINICAL USE:** This application demonstrates "
    "a machine learning model trained on the public Indian Liver Patient Dataset (ILPD). "
    "It is **not** a clinically validated diagnostic tool and must **never** be used to "
    "diagnose patients or guide medical decisions. Displayed probabilities are statistical "
    "model risk scores, not clinically calibrated diagnostic probabilities."
)

SAMPLE_CSV_TEMPLATE = """age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio
62,Male,10.9,5.5,699,64,100,7.5,3.2,0.74
29,Female,0.7,0.1,162,20,21,7.0,3.8,1.18
46,Male,1.8,0.7,208,19,14,7.6,4.4,1.30
"""


def _headers(api_key: str) -> Dict[str, str]:
    return {"X-API-Key": api_key.strip()} if api_key.strip() else {}


def _check_health(api_url: str) -> Dict[str, Any] | None:
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def _fetch_model_info(api_url: str, api_key: str) -> tuple[int, Dict[str, Any]]:
    try:
        resp = requests.get(f"{api_url}/model-info", headers=_headers(api_key), timeout=10)
        return resp.status_code, resp.json()
    except requests.RequestException as exc:
        return 0, {"detail": f"Cannot reach API at {api_url}: {exc}"}


def _build_probability_gauge(probability: float, threshold: float, risk_category: str) -> go.Figure:
    color_map = {"Low": "#2E7D32", "Moderate": "#ED6C02", "High": "#C62828"}
    bar_color = color_map.get(risk_category, "#1976D2")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100.0, 2),
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": f"Model Risk Score — {risk_category} Risk Band", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 35], "color": "#E8F5E9"},
                    {"range": [35, 65], "color": "#FFF3E0"},
                    {"range": [65, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100.0,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=55, b=10))
    return fig


def _build_shap_bar_chart(contributions: list[dict[str, Any]]) -> go.Figure:
    df_shap = pd.DataFrame(contributions).sort_values("abs_impact", ascending=True)
    colors = [
        "#D32F2F" if d == "increases_risk" else "#1976D2"
        for d in df_shap["direction"]
    ]
    labels = [
        f"{row['feature']} (val={row['raw_value']})"
        for _, row in df_shap.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            x=df_shap["shap_value"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Individual Feature Contributions (SHAP Values)",
        xaxis_title="SHAP Value (Negative = Lower Risk Score | Positive = Higher Risk Score)",
        yaxis_title="",
        height=360,
        margin=dict(l=20, r=40, t=50, b=30),
    )
    fig.add_vline(x=0, line_width=1, line_color="#333333")
    return fig


# ------------------------------------------------------------------------------
# Sidebar: Backend API Connection & Authentication
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("API Connection")
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL).rstrip("/")
    api_key = st.text_input(
        "API Key (`X-API-Key`)",
        value=DEFAULT_API_KEY,
        type="password",
        help="Must match the LIVER_API_KEY environment variable configured on the FastAPI server.",
    )

    health = _check_health(api_url)
    if health and health.get("model_loaded"):
        st.success(
            f"API Connected: **{health['status'].upper()}**\n\n"
            f"Model: `{health.get('model_name')}` (v{health.get('model_version')})"
        )
    elif health:
        st.warning("API reachable, but model artifact is not loaded.")
    else:
        st.error(f"Cannot reach FastAPI backend at `{api_url}`.")

    st.divider()
    st.caption(
        "Need Swagger UI? Open "
        f"[`{api_url}/docs`]({api_url}/docs) in your browser."
    )


# ------------------------------------------------------------------------------
# Main Header & Disclaimer
# ------------------------------------------------------------------------------
st.title("🩺 Liver Disease Prediction System")
st.warning(MEDICAL_DISCLAIMER)

tab_single, tab_batch, tab_info = st.tabs(
    ["Individual Patient Prediction", "Batch CSV Prediction", "Model Metadata & Global Importance"]
)

# ------------------------------------------------------------------------------
# Tab 1: Individual Patient Prediction + SHAP Explainability
# ------------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter Patient Liver Function Panel Attributes")

    preset_col1, preset_col2, _ = st.columns([1, 1, 2])
    if preset_col1.button("Load Elevated LFT Sample", use_container_width=True):
        st.session_state["preset"] = {
            "age": 62,
            "gender": "Male",
            "tb": 10.9,
            "db": 5.5,
            "alkphos": 699.0,
            "sgpt": 64.0,
            "sgot": 100.0,
            "tp": 7.5,
            "alb": 3.2,
            "ag": 0.74,
        }
    if preset_col2.button("Load Normal-Range Sample", use_container_width=True):
        st.session_state["preset"] = {
            "age": 29,
            "gender": "Female",
            "tb": 0.7,
            "db": 0.1,
            "alkphos": 162.0,
            "sgpt": 20.0,
            "sgot": 21.0,
            "tp": 7.0,
            "alb": 3.8,
            "ag": 1.18,
        }

    p = st.session_state.get(
        "preset",
        {
            "age": 45,
            "gender": "Male",
            "tb": 1.2,
            "db": 0.4,
            "alkphos": 210.0,
            "sgpt": 38.0,
            "sgot": 45.0,
            "tp": 6.8,
            "alb": 3.3,
            "ag": 0.95,
        },
    )

    with st.form("single_patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=int(p["age"]), step=1)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0 if p["gender"] == "Male" else 1)
            tb = st.number_input("Total Bilirubin — TB (mg/dL) [Ref: 0.1–1.2]", min_value=0.05, max_value=100.0, value=float(p["tb"]), step=0.1, format="%.2f")
            db = st.number_input("Direct Bilirubin — DB (mg/dL) [Ref: 0.0–0.3]", min_value=0.0, max_value=50.0, value=float(p["db"]), step=0.1, format="%.2f")
        with c2:
            alkphos = st.number_input("Alkaline Phosphotase — ALP (IU/L) [Ref: 44–147]", min_value=10.0, max_value=5000.0, value=float(p["alkphos"]), step=1.0)
            sgpt = st.number_input("Alamine Aminotransferase — ALT/SGPT (IU/L) [Ref: 7–56]", min_value=1.0, max_value=5000.0, value=float(p["sgpt"]), step=1.0)
            sgot = st.number_input("Aspartate Aminotransferase — AST/SGOT (IU/L) [Ref: 10–40]", min_value=1.0, max_value=10000.0, value=float(p["sgot"]), step=1.0)
        with c3:
            tp = st.number_input("Total Proteins — TP (g/dL) [Ref: 6.0–8.3]", min_value=1.0, max_value=20.0, value=float(p["tp"]), step=0.1, format="%.2f")
            alb = st.number_input("Albumin — ALB (g/dL) [Ref: 3.5–5.0]", min_value=0.5, max_value=12.0, value=float(p["alb"]), step=0.1, format="%.2f")
            ag = st.number_input("A/G Ratio (dimensionless) [Ref: 1.0–2.5]", min_value=0.1, max_value=10.0, value=float(p["ag"]), step=0.05, format="%.2f")

        include_shap = st.checkbox("Include SHAP Feature Attribution Explanation", value=True)
        submitted = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)

    if submitted:
        if not api_key.strip():
            st.error("Please enter your API Key (`X-API-Key`) in the sidebar before submitting.")
        else:
            payload = {
                "age": int(age),
                "gender": gender,
                "total_bilirubin": float(tb),
                "direct_bilirubin": float(db),
                "alkaline_phosphotase": float(alkphos),
                "alamine_aminotransferase": float(sgpt),
                "aspartate_aminotransferase": float(sgot),
                "total_proteins": float(tp),
                "albumin": float(alb),
                "albumin_and_globulin_ratio": float(ag),
            }
            endpoint = "/explain" if include_shap else "/predict"
            try:
                resp = requests.post(
                    f"{api_url}{endpoint}",
                    json=payload,
                    headers=_headers(api_key),
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Predicted Class", f"{data['prediction']} — {data['prediction_label']}")
                    m2.metric("Model Probability Score", f"{data['probability'] * 100:.2f}%")
                    m3.metric("Risk Category", data["risk_category"])
                    m4.metric("Decision Threshold", f"{data['decision_threshold']:.2f}")

                    st.caption(
                        f"Prediction ID: `{data['prediction_id']}` | Model Version: `v{data['model_version']}`"
                    )

                    g_col, s_col = st.columns([1, 1.35])
                    with g_col:
                        st.plotly_chart(
                            _build_probability_gauge(
                                data["probability"],
                                data["decision_threshold"],
                                data["risk_category"],
                            ),
                            use_container_width=True,
                        )
                    with s_col:
                        if include_shap and "explanation" in data:
                            exp = data["explanation"]
                            st.plotly_chart(
                                _build_shap_bar_chart(exp["top_contributing_features"]),
                                use_container_width=True,
                            )
                            st.info(f"**Attribution Note ({exp['explainer_type']}):** {exp['limitations_note']}")
                elif resp.status_code == 401:
                    st.error("Authentication failed (HTTP 401). Check your `X-API-Key` in the sidebar.")
                else:
                    st.error(f"API returned HTTP {resp.status_code}: {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Request to FastAPI backend failed: {exc}")


# ------------------------------------------------------------------------------
# Tab 2: Batch CSV Predictions & Download
# ------------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Predictions via CSV Upload")
    st.markdown(
        "Upload a CSV file using either **canonical API column names** "
        "(`age, gender, total_bilirubin, ...`) or **raw ILPD column names** "
        "(`Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio`)."
    )
    st.download_button(
        label="Download Sample CSV Template",
        data=SAMPLE_CSV_TEMPLATE.encode("utf-8"),
        file_name="ilpd_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])
    if uploaded_file is not None:
        if st.button("Run Batch Predictions", type="primary"):
            if not api_key.strip():
                st.error("Please enter your API Key (`X-API-Key`) in the sidebar.")
            else:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    resp = requests.post(
                        f"{api_url}/predict/batch",
                        files=files,
                        headers=_headers(api_key),
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        batch_data = resp.json()
                        preds_df = pd.DataFrame(batch_data["predictions"])

                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Total Rows", batch_data["total_rows"])
                        k2.metric("Valid Predictions", batch_data["valid_count"])
                        k3.metric("Invalid Rows", batch_data["invalid_count"])
                        high_cnt = int((preds_df["risk_category"] == "High").sum()) if not preds_df.empty else 0
                        k4.metric("High Risk Count", high_cnt)

                        st.dataframe(preds_df, use_container_width=True)

                        # Merge with original uploaded rows for convenient export
                        orig_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                        orig_df["row_index"] = orig_df.index
                        export_df = orig_df.merge(preds_df, on="row_index", how="left")
                        st.download_button(
                            label="Download Batch Predictions CSV",
                            data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"predictions_{batch_data['batch_id'][:8]}.csv",
                            mime="text/csv",
                        )

                        if batch_data["invalid_records"]:
                            st.warning(f"{batch_data['invalid_count']} row(s) failed validation and were excluded:")
                            st.dataframe(pd.DataFrame(batch_data["invalid_records"]), use_container_width=True)
                    elif resp.status_code == 401:
                        st.error("Authentication failed (HTTP 401). Check your `X-API-Key`.")
                    else:
                        st.error(f"Batch API returned HTTP {resp.status_code}: {resp.text}")
                except requests.RequestException as exc:
                    st.error(f"Batch request failed: {exc}")


# ------------------------------------------------------------------------------
# Tab 3: Model Metadata & Global SHAP Importance
# ------------------------------------------------------------------------------
with tab_info:
    st.subheader("Loaded Model Metadata (`GET /model-info`)")
    if not api_key.strip():
        st.info("Enter your `X-API-Key` in the sidebar to inspect authenticated `/model-info` metadata.")
    else:
        status_code, info = _fetch_model_info(api_url, api_key)
        if status_code == 200:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Name", info["model_name"])
            c2.metric("Model Version", f"v{info['model_version']}")
            c3.metric("Decision Threshold", f"{info['decision_threshold']:.2f}")
            c4.metric("Calibration", info["calibration_method"])

            st.markdown("### Global Feature Importance (Mean |SHAP|)")
            gfi = info.get("global_feature_importance", [])
            if gfi:
                gfi_df = pd.DataFrame(gfi).sort_values("mean_abs_shap", ascending=True)
                fig_g = go.Figure(
                    go.Bar(
                        x=gfi_df["mean_abs_shap"],
                        y=gfi_df["feature"],
                        orientation="h",
                        marker_color="#2C7BB6",
                    )
                )
                fig_g.update_layout(
                    xaxis_title="Mean |SHAP value|",
                    height=340,
                    margin=dict(l=20, r=20, t=20, b=30),
                )
                st.plotly_chart(fig_g, use_container_width=True)

            st.markdown("### Required Input Features & Medical Units")
            st.dataframe(pd.DataFrame(info["required_features"]), use_container_width=True)

            with st.expander("Raw Evaluation & Dataset Metadata JSON"):
                st.json(
                    {
                        "dataset_info": info["dataset_info"],
                        "evaluation_metrics": info["evaluation_metrics"],
                        "risk_category_bands": info["risk_category_bands"],
                        "target_encoding": info["target_encoding"],
                    }
                )
        elif status_code == 401:
            st.error("Authentication failed (HTTP 401). Enter the valid API key in the sidebar.")
        else:
            st.error(f"Failed to fetch model metadata (HTTP {status_code}): {info}")
