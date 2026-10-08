"""
TwinVerse – A Human Digital Twin Framework for Personalized Disease Prediction
Integrated Healthcare AI Research Dashboard
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Ensure models directory is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from clinical_model import predict_patient_risk, load_clinical_model
from xray_inference import predict_xray, load_xray_model, preprocess_xray
from explainability import generate_gradcam, overlay_heatmap, get_clinical_feature_importance
from fusion_model import fuse_predictions, get_risk_category
from dynamic_state import (
    create_patient_state,
    add_state_to_history,
    history_to_dataframe,
    calculate_risk_trend,
    get_risk_change,
)
import twinverse_core as core

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="TwinVerse – Human Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern Medical Dashboard CSS (Dark Theme)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        background-color: #0b111e;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* Professional Card styling */
    .tv-card {
        background: #111a2e;
        border: 1px solid rgba(0, 210, 196, 0.16);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .tv-card-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .tv-card-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #00d2c4;
        margin-bottom: 0.2rem;
    }
    .tv-card-desc {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Risk Badges */
    .tv-badge-low {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 9999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .tv-badge-mod {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 9999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .tv-badge-high {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 9999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Pipeline block */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        background: #111a2e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin: 1.2rem 0;
    }
    .pipeline-step {
        text-align: center;
        flex: 1;
        min-width: 140px;
        padding: 0.5rem;
    }
    .pipeline-step-title {
        color: #00d2c4;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .pipeline-step-sub {
        color: #94a3b8;
        font-size: 0.8rem;
    }
    .pipeline-arrow {
        color: #64748b;
        font-size: 1.3rem;
        font-weight: 700;
    }

    /* Streamlit widget tweaks */
    div[data-testid="stMetricValue"] {
        color: #00d2c4 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #0c1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d2c4 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

if "multimodal_result" not in st.session_state:
    st.session_state.multimodal_result = None

if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

if "current_twin" not in st.session_state:
    st.session_state.current_twin = None


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def render_badge(category: str):
    if "HIGH" in category.upper():
        return f'<span class="tv-badge-high">{category}</span>'
    elif "MODERATE" in category.upper():
        return f'<span class="tv-badge-mod">{category}</span>'
    return f'<span class="tv-badge-low">{category}</span>'


def get_risk_color(score: float):
    if score >= 70.0:
        return "#ef4444"
    elif score >= 40.0:
        return "#f59e0b"
    return "#10b981"


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.6rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
        <h2 style="font-size: 1.55rem; margin: 0; color: #00d2c4; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            🧬 TwinVerse
        </h2>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 0.2rem 0 0 0; font-weight: 500;">
            Human Digital Twin
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Patient Digital Twin",
        "Multimodal Analysis",
        "What-If Simulation",
        "Digital Twin Timeline",
        "Clinical Intelligence",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Active patient indicator in sidebar
if st.session_state.current_patient:
    p_name = st.session_state.current_patient.get("name", "Active Patient")
    p_age = int(st.session_state.current_patient.get("age", 0))
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(0, 210, 196, 0.08); border: 1px solid rgba(0, 210, 196, 0.2); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Active Patient</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">{p_name} ({p_age}y)</div>
            <div style="font-size: 0.78rem; color: #00d2c4;">Visits Recorded: {len(st.session_state.patient_history)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase;">Digital Twin Status</div>
            <div style="font-size: 0.88rem; color: #94a3b8;">No patient profile initialized</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.sidebar.button("Reset Digital Twin", use_container_width=True):
    st.session_state.patient_history = []
    st.session_state.multimodal_result = None
    st.session_state.current_patient = None
    st.session_state.current_twin = None
    st.rerun()


# =========================================================
# 1. HOME PAGE
# =========================================================
if nav_selection == "Home":
    st.title("TwinVerse")
    st.markdown(
        "<h3 style='color: #00d2c4; font-size: 1.25rem; font-weight: 500; margin-top: -0.8rem; margin-bottom: 1rem;'>"
        "A Human Digital Twin Framework for Personalized Disease Prediction</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem; line-height: 1.6; max-width: 900px;'>"
        "TwinVerse combines structured clinical information and chest X-ray analysis to generate a multimodal "
        "patient risk assessment and maintain a dynamic digital representation of the patient's health state."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # Three professional metric cards
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown(
            """
            <div class="tv-card">
                <div class="tv-card-header">Clinical Analysis</div>
                <div class="tv-card-value">Status: Ready</div>
                <div class="tv-card-desc">11-Feature Random Forest Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            """
            <div class="tv-card">
                <div class="tv-card-header">Imaging Analysis</div>
                <div class="tv-card-value">Status: Ready</div>
                <div class="tv-card-desc">Deep Radiograph Cardiomegaly Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"""
            <div class="tv-card">
                <div class="tv-card-header">Digital Twin</div>
                <div class="tv-card-value">Status: Active</div>
                <div class="tv-card-desc">{len(st.session_state.patient_history)} Observation Encounters Logged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # "How TwinVerse Works" section
    st.markdown("<h3 style='margin-top: 1.8rem;'>How TwinVerse Works</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-step">
                <div class="pipeline-step-title">Patient Data</div>
                <div class="pipeline-step-sub">Physiological vitals & chest radiographs</div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">Clinical + X-Ray Analysis</div>
                <div class="pipeline-step-sub">Biomarker scoring & CNN inference</div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">Multimodal Fusion</div>
                <div class="pipeline-step-sub">Probability-level late fusion</div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">Digital Twin Insights</div>
                <div class="pipeline-step-sub">Longitudinal trajectory & simulation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards (2x3 grid)
    st.markdown("<h3 style='margin-top: 1.8rem;'>Core Capabilities</h3>", unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">Clinical Risk Prediction</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Calibrated prediction of cardiac event probability utilizing 11 clinical features
                    including renal markers, ejection fraction, and systemic risk factors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">Chest X-Ray Analysis</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Neural radiographic evaluation screening for cardiomegaly and structural anatomical
                    abnormalities via convolutional deep learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fcol2:
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">Explainable AI</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Dual-modality transparency through Grad-CAM saliency maps of cardiac regions and
                    quantitative Random Forest feature importance rankings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">Dynamic Digital Twin</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Longitudinal state tracking that archives visit history, maps patient health trajectories,
                    and monitors automated risk directionality.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fcol3:
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">What-If Simulation</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Interactive scenario modeling enabling physicians to explore how biomarker changes
                    (e.g., ejection fraction, creatinine) impact predicted risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="tv-card">
                <h4 style="color: #00d2c4; font-size: 1.05rem; margin-top: 0;">Clinical Intelligence</h4>
                <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5;">
                    Consolidated patient synthesis integrating imaging insights, clinical drivers,
                    and longitudinal trends into actionable clinical summaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 2. PATIENT DIGITAL TWIN PAGE
# =========================================================
elif nav_selection == "Patient Digital Twin":
    st.title("Patient Digital Twin")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem;'>"
        "Create a digital representation of the patient's current clinical health state and generate an ML-based risk prediction."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("digital_twin_form"):
        st.subheader("Patient Information")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            p_name = st.text_input("Patient Name", value="Eleanor Vance")
        with pcol2:
            p_age = st.number_input("Age", min_value=18, max_value=110, value=65, step=1)
        with pcol3:
            p_sex = st.selectbox("Sex", ["Male", "Female"], index=0)

        st.subheader("Clinical Parameters")
        ccol1, ccol2, ccol3 = st.columns(3)

        with ccol1:
            ef = st.number_input(
                "Ejection Fraction (%)",
                min_value=10.0,
                max_value=85.0,
                value=30.0,
                step=1.0,
                help="Heart pumping efficiency (normal: 55-70%)",
            )
            sc = st.number_input(
                "Serum Creatinine (mg/dL)",
                min_value=0.3,
                max_value=12.0,
                value=1.8,
                step=0.1,
                help="Kidney function indicator (normal: 0.7-1.3)",
            )
            ss = st.number_input(
                "Serum Sodium (mEq/L)",
                min_value=105.0,
                max_value=160.0,
                value=134.0,
                step=1.0,
                help="Electrolyte balance indicator (normal: 135-145)",
            )
            platelets = st.number_input(
                "Platelets (count/mcL)",
                min_value=25000.0,
                max_value=900000.0,
                value=263000.0,
                step=1000.0,
            )

        with ccol2:
            cpk = st.number_input(
                "Creatinine Phosphokinase (mcg/L)",
                min_value=15.0,
                max_value=9000.0,
                value=582.0,
                step=10.0,
                help="Enzyme marker of muscular or cardiac stress",
            )
            hbp = st.selectbox(
                "High Blood Pressure",
                ["No", "Yes"],
                index=1,
                help="Hypertension history",
            )
            anaemia = st.selectbox(
                "Anaemia",
                ["No", "Yes"],
                index=0,
                help="Reduction in red blood cells or hemoglobin",
            )

        with ccol3:
            diabetes = st.selectbox(
                "Diabetes",
                ["No", "Yes"],
                index=0,
                help="Pre-existing diabetes diagnosis",
            )
            smoking = st.selectbox(
                "Smoking",
                ["No", "Yes"],
                index=0,
                help="Active or recent tobacco smoking",
            )

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        generate_btn = st.form_submit_button("Generate Digital Twin", use_container_width=True)

    if generate_btn:
        try:
            # 1. Create patient_data dictionary
            sex_numeric = 1 if p_sex == "Male" else 0
            hbp_numeric = 1 if hbp == "Yes" else 0
            anaemia_numeric = 1 if anaemia == "Yes" else 0
            diabetes_numeric = 1 if diabetes == "Yes" else 0
            smoking_numeric = 1 if smoking == "Yes" else 0

            patient_data = {
                "age": float(p_age),
                "anaemia": anaemia_numeric,
                "creatinine_phosphokinase": float(cpk),
                "diabetes": diabetes_numeric,
                "ejection_fraction": float(ef),
                "high_blood_pressure": hbp_numeric,
                "platelets": float(platelets),
                "serum_creatinine": float(sc),
                "serum_sodium": float(ss),
                "sex": sex_numeric,
                "smoking": smoking_numeric,
            }

            # 2. Call predict_patient_risk
            risk_probability, prediction = predict_patient_risk(patient_data)

            # 3. Calculate risk percentage
            risk_pct = round(risk_probability * 100.0 if risk_probability <= 1.0 else risk_probability, 2)
            risk_cat = get_risk_category(risk_pct)

            # 4. Create a digital twin state
            state = create_patient_state(
                patient_name=p_name,
                patient_data=patient_data,
                clinical_risk=risk_pct,
                fused_risk=risk_pct,
            )

            # 5. Save the state to patient_history
            st.session_state.patient_history = add_state_to_history(state, st.session_state.patient_history)
            st.session_state.current_twin = state
            st.session_state.current_patient = {
                "name": p_name,
                "age": p_age,
                "sex": p_sex,
                "data": patient_data,
                "risk_pct": risk_pct,
                "category": risk_cat,
            }

            st.success(f"✓ Digital Twin state generated and saved to encounter history (Visit {state['visit']}).")

        except Exception as e:
            st.error(f"Error generating Digital Twin: {str(e)}")

    # Display results if a digital twin exists
    if st.session_state.current_twin:
        t = st.session_state.current_twin
        cp = st.session_state.current_patient

        st.markdown("<h3 style='margin-top: 1.5rem;'>Digital Twin Assessment</h3>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.2, 2])
        with res_col1:
            color = get_risk_color(t["risk_score"])
            cat = get_risk_category(t["risk_score"])
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Predicted Risk</div>
                    <div style="font-size: 2.6rem; font-weight: 800; color: {color}; margin: 0.2rem 0;">
                        {t['risk_score']:.2f}%
                    </div>
                    <div style="margin-top: 0.4rem;">
                        {render_badge(cat)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f"""
                <div class="tv-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div class="tv-card-header">Clinical Risk Gauge</div>
                    <div style="background: rgba(255,255,255,0.08); border-radius: 9999px; height: 16px; width: 100%; overflow: hidden; margin: 0.8rem 0;">
                        <div style="background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); width: {min(100.0, max(5.0, t['risk_score']))}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b;">
                        <span>0% (Low Risk)</span>
                        <span>40% (Moderate Risk)</span>
                        <span>70%+ (High Risk)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Current Patient Health State cards
        st.markdown("<h4>Current Patient Health State</h4>", unsafe_allow_html=True)
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Ejection Fraction</div>
                    <div class="tv-card-value">{t['ejection_fraction']:.1f}%</div>
                    <div class="tv-card-desc">{"Reduced pumping function" if t['ejection_fraction'] <= 40 else "Normal systolic function"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hcol2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Creatinine</div>
                    <div class="tv-card-value">{t['serum_creatinine']:.2f} mg/dL</div>
                    <div class="tv-card-desc">{"Elevated renal load" if t['serum_creatinine'] >= 1.4 else "Normal filtration"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hcol3:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Sodium</div>
                    <div class="tv-card-value">{t['serum_sodium']:.1f} mEq/L</div>
                    <div class="tv-card-desc">{"Mild hyponatremia" if t['serum_sodium'] < 135 else "Electrolytes in range"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Digital Twin Summary
        st.markdown("<h4>Digital Twin Summary</h4>", unsafe_allow_html=True)
        cd = t.get("clinical_data", {})
        st.markdown(
            f"""
            <div class="tv-card">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                    <div><span style="color:#64748b; font-size:0.8rem;">NAME:</span> <strong style="color:#f8fafc;">{t['name']}</strong></div>
                    <div><span style="color:#64748b; font-size:0.8rem;">AGE:</span> <strong style="color:#f8fafc;">{int(t['age'])}</strong></div>
                    <div><span style="color:#64748b; font-size:0.8rem;">SEX:</span> <strong style="color:#f8fafc;">{t['sex']}</strong></div>
                    <div><span style="color:#64748b; font-size:0.8rem;">DIABETES:</span> <strong style="color:#f8fafc;">{'Yes' if cd.get('diabetes') == 1 else 'No'}</strong></div>
                    <div><span style="color:#64748b; font-size:0.8rem;">HYPERTENSION:</span> <strong style="color:#f8fafc;">{'Yes' if cd.get('high_blood_pressure') == 1 else 'No'}</strong></div>
                    <div><span style="color:#64748b; font-size:0.8rem;">SMOKING:</span> <strong style="color:#f8fafc;">{'Yes' if cd.get('smoking') == 1 else 'No'}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 3. MULTIMODAL ANALYSIS PAGE
# =========================================================
elif nav_selection == "Multimodal Analysis":
    st.title("Multimodal Medical Analysis")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem;'>"
        "TwinVerse combines structured clinical information and chest X-ray analysis to generate an integrated risk assessment."
        "</p>",
        unsafe_allow_html=True,
    )

    # Pre-populate defaults from active patient if available
    curr_data = st.session_state.current_twin.get("clinical_data", {}) if st.session_state.current_twin else {}

    st.subheader("Clinical Patient Data")
    m_ccol1, m_ccol2, m_ccol3 = st.columns(3)
    with m_ccol1:
        m_age = st.number_input("Age", 18, 110, int(curr_data.get("age", 65)))
        m_ef = st.number_input("Ejection Fraction (%)", 10.0, 85.0, float(curr_data.get("ejection_fraction", 30.0)), step=1.0)
        m_sc = st.number_input("Serum Creatinine (mg/dL)", 0.3, 12.0, float(curr_data.get("serum_creatinine", 1.8)), step=0.1)
        m_ss = st.number_input("Serum Sodium (mEq/L)", 105.0, 160.0, float(curr_data.get("serum_sodium", 134.0)), step=1.0)
    with m_ccol2:
        m_cpk = st.number_input("Creatinine Phosphokinase", 15.0, 9000.0, float(curr_data.get("creatinine_phosphokinase", 582.0)), step=10.0)
        m_platelets = st.number_input("Platelets", 25000.0, 900000.0, float(curr_data.get("platelets", 263000.0)), step=1000.0)
        m_hbp = st.selectbox("High Blood Pressure", ["No", "Yes"], index=1 if curr_data.get("high_blood_pressure", 1) == 1 else 0)
        m_anaemia = st.selectbox("Anaemia", ["No", "Yes"], index=1 if curr_data.get("anaemia", 0) == 1 else 0)
    with m_ccol3:
        m_diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=1 if curr_data.get("diabetes", 0) == 1 else 0)
        m_smoking = st.selectbox("Smoking", ["No", "Yes"], index=1 if curr_data.get("smoking", 0) == 1 else 0)
        m_sex = st.selectbox("Sex", ["Male", "Female"], index=0 if curr_data.get("sex", 1) == 1 else 1)

    st.subheader("Chest X-Ray Analysis")
    xcol1, xcol2 = st.columns([1.5, 1])

    with xcol1:
        xray_file = st.file_uploader("Upload Chest Radiograph", type=["png", "jpg", "jpeg"])

        # Also provide preset sample radiographs for instant convenience during review
        preset_choice = st.selectbox(
            "Or select a preset demonstration radiograph:",
            [
                "None (Use Uploaded File)",
                "Demo Case (High Cardiomegaly Probability)",
                "Synthetic Abnormal (Cardiomegaly Finding)",
                "Synthetic Normal (Clear Lung Fields)",
            ],
            index=1 if xray_file is None else 0,
        )

    # Determine which image source to use
    selected_image = None
    if xray_file is not None:
        selected_image = Image.open(xray_file)
    elif preset_choice == "Demo Case (High Cardiomegaly Probability)":
        p_path = os.path.join(BASE_DIR, "assets", "demo_patient_xray.png")
        if os.path.exists(p_path):
            selected_image = Image.open(p_path)
    elif preset_choice == "Synthetic Abnormal (Cardiomegaly Finding)":
        p_path = os.path.join(BASE_DIR, "data", "xray_synth", "abnormal", "abnormal_0001.png")
        if os.path.exists(p_path):
            selected_image = Image.open(p_path)
    elif preset_choice == "Synthetic Normal (Clear Lung Fields)":
        p_path = os.path.join(BASE_DIR, "data", "xray_synth", "normal", "normal_0001.png")
        if os.path.exists(p_path):
            selected_image = Image.open(p_path)

    with xcol2:
        if selected_image is not None:
            st.markdown(
                """
                <div class="tv-card" style="padding: 0.8rem; text-align: center;">
                    <div class="tv-card-header">Radiograph Preview</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(selected_image, width=220)
        else:
            st.info("Upload or select a radiograph to enable multimodal processing.")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    run_mm_btn = st.button("Run Multimodal Analysis", use_container_width=True)

    if run_mm_btn:
        if selected_image is None:
            st.error("Please provide a chest X-ray image (upload or select preset) before running analysis.")
        else:
            try:
                with st.spinner("Processing clinical biomarkers and performing deep X-ray inference..."):
                    # 1. Save uploaded image temporarily if needed
                    temp_dir = tempfile.gettempdir()
                    temp_img_path = os.path.join(temp_dir, "twinverse_current_xray.png")
                    selected_image.save(temp_img_path)

                    # 2. Create clinical_data
                    clinical_data = {
                        "age": float(m_age),
                        "anaemia": 1 if m_anaemia == "Yes" else 0,
                        "creatinine_phosphokinase": float(m_cpk),
                        "diabetes": 1 if m_diabetes == "Yes" else 0,
                        "ejection_fraction": float(m_ef),
                        "high_blood_pressure": 1 if m_hbp == "Yes" else 0,
                        "platelets": float(m_platelets),
                        "serum_creatinine": float(m_sc),
                        "serum_sodium": float(m_ss),
                        "sex": 1 if m_sex == "Male" else 0,
                        "smoking": 1 if m_smoking == "Yes" else 0,
                    }

                    # 3. Call predict_patient_risk
                    c_prob, c_pred = predict_patient_risk(clinical_data)
                    c_risk_pct = round(c_prob * 100.0 if c_prob <= 1.0 else c_prob, 2)

                    # 4. Call predict_xray
                    x_prob, x_pred = predict_xray(temp_img_path)
                    x_risk_pct = round(x_prob * 100.0 if x_prob <= 1.0 else x_prob, 2)

                    # 5. Call generate_gradcam
                    model = load_xray_model()
                    tensor, img_gray = preprocess_xray(temp_img_path)
                    heatmap, pred_cls, probs = generate_gradcam(model, tensor)
                    cam_overlay = overlay_heatmap(img_gray, heatmap)

                    # 6. Call fuse_predictions
                    fused_risk_pct = fuse_predictions(c_risk_pct, x_risk_pct, w_clinical=0.50, w_xray=0.50)

                    # 7. Call get_risk_category
                    overall_category = get_risk_category(fused_risk_pct)

                    # 8. Save result into session_state.multimodal_result
                    mm_result = {
                        "clinical_risk": c_risk_pct,
                        "xray_risk": x_risk_pct,
                        "fused_risk": fused_risk_pct,
                        "category": overall_category,
                        "is_abnormal": x_pred == 1,
                        "original_image": img_gray,
                        "gradcam_image": cam_overlay,
                        "clinical_data": clinical_data,
                    }
                    st.session_state.multimodal_result = mm_result

                    # 9. Save the fused prediction into the Digital Twin history
                    patient_name = st.session_state.current_patient.get("name", "Demo Patient") if st.session_state.current_patient else "Demo Patient"
                    new_state = create_patient_state(
                        patient_name=patient_name,
                        patient_data=clinical_data,
                        clinical_risk=c_risk_pct,
                        xray_risk=x_risk_pct,
                        fused_risk=fused_risk_pct,
                    )
                    st.session_state.patient_history = add_state_to_history(new_state, st.session_state.patient_history)
                    st.session_state.current_twin = new_state

                    st.success("Multimodal fusion assessment successfully computed!")

            except Exception as e:
                st.error(f"Multimodal analysis encountered an issue: {str(e)}")

    # ---------------------------------------------------------
    # MULTIMODAL RESULT UI
    # ---------------------------------------------------------
    if st.session_state.multimodal_result:
        res = st.session_state.multimodal_result

        st.markdown("<h3 style='margin-top: 2rem;'>Multimodal Risk Assessment</h3>", unsafe_allow_html=True)

        # Three large result cards
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Clinical Risk</div>
                    <div class="tv-card-value">{res['clinical_risk']:.2f}%</div>
                    <div class="tv-card-desc">Random Forest (11 Biomarkers)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rcol2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">X-Ray Risk</div>
                    <div class="tv-card-value">{res['xray_risk']:.2f}%</div>
                    <div class="tv-card-desc">Cardiomegaly Detection Probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rcol3:
            st.markdown(
                f"""
                <div class="tv-card" style="border-color: rgba(0, 210, 196, 0.45);">
                    <div class="tv-card-header" style="color: #00d2c4;">Fused Risk</div>
                    <div class="tv-card-value" style="font-size: 2.2rem;">{res['fused_risk']:.2f}%</div>
                    <div class="tv-card-desc">Late Probability-Level Fusion (50:50)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Overall TwinVerse Risk with Risk Category and visual progress bar
        st.markdown("<h4>Overall TwinVerse Risk</h4>", unsafe_allow_html=True)
        color = get_risk_color(res["fused_risk"])
        st.markdown(
            f"""
            <div class="tv-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Integrated Patient State: </span>
                        {render_badge(res['category'])}
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: {color};">
                        {res['fused_risk']:.2f}%
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.08); border-radius: 9999px; height: 14px; width: 100%; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); width: {min(100.0, max(5.0, res['fused_risk']))}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # X-Ray Explainability (Side-by-side)
        st.markdown("<h4>X-Ray Explainability</h4>", unsafe_allow_html=True)
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown(
                """
                <div class="tv-card" style="text-align: center;">
                    <div class="tv-card-header">Original X-Ray</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res["original_image"], use_container_width=True)

        with ecol2:
            st.markdown(
                """
                <div class="tv-card" style="text-align: center;">
                    <div class="tv-card-header">AI Attention / Grad-CAM</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(res["gradcam_image"], use_container_width=True)

        st.caption(
            "Grad-CAM visualizes convolutional feature attention maps and highlights thoracic regions "
            "influencing the model's estimate; it does not replace formal radiological examination."
        )

        # Integrated Interpretation
        st.markdown("<h4>Integrated Interpretation</h4>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="tv-card">
                <div style="line-height: 1.6; color: #cbd5e1; font-size: 0.95rem;">
                    • <strong>Clinical Model Probability:</strong> {res['clinical_risk']:.2f}% (derived from 11 clinical features including ejection fraction and serum creatinine)<br>
                    • <strong>X-Ray Model Probability:</strong> {res['xray_risk']:.2f}% (derived from thoracic radiograph feature representations)<br>
                    • <strong>Fused Multimodal Probability:</strong> {res['fused_risk']:.2f}% (balanced late probability-level integration)<br>
                    • <strong>Overall Risk Category:</strong> {res['category']}<br><br>
                    <em>TwinVerse balances laboratory biomarkers with imaging evidence so that confident predictions in one modality are contextualized by the other, providing a balanced, holistic evaluation of patient health.</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 4. WHAT-IF SIMULATION
# =========================================================
elif nav_selection == "What-If Simulation":
    st.title("What-If Health Simulation")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem;'>"
        "Modify selected clinical parameters to observe how the trained model's estimated risk changes."
        "</p>",
        unsafe_allow_html=True,
    )

    # Base patient defaults
    curr = st.session_state.current_twin.get("clinical_data", {}) if st.session_state.current_twin else {}

    cur_col, sim_col = st.columns(2)

    with cur_col:
        st.markdown(
            """
            <div class="tv-card-header" style="color: #00d2c4; font-size: 1rem; margin-bottom: 0.8rem;">
                Current Patient Baseline
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_age = st.number_input("Age", 18, 110, int(curr.get("age", 65)), key="wi_age")
        c_ef = st.number_input("Ejection Fraction (%)", 10.0, 85.0, float(curr.get("ejection_fraction", 30.0)), step=1.0, key="wi_ef")
        c_sc = st.number_input("Serum Creatinine (mg/dL)", 0.3, 12.0, float(curr.get("serum_creatinine", 1.8)), step=0.1, key="wi_sc")
        c_ss = st.number_input("Serum Sodium (mEq/L)", 105.0, 160.0, float(curr.get("serum_sodium", 134.0)), step=1.0, key="wi_ss")
        c_smoking = st.selectbox("Smoking", ["No", "Yes"], index=1 if curr.get("smoking", 0) == 1 else 0, key="wi_smk")
        c_diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=1 if curr.get("diabetes", 0) == 1 else 0, key="wi_dia")

    with sim_col:
        st.markdown(
            """
            <div class="tv-card-header" style="color: #38bdf8; font-size: 1rem; margin-bottom: 0.8rem;">
                Simulated Parameters
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color: #64748b; font-size: 0.85rem;'>Adjust target values to simulate clinical adjustments:</p>",
            unsafe_allow_html=True,
        )
        sim_ef = st.slider("Simulated Ejection Fraction (%)", 10.0, 80.0, 50.0, step=1.0, help="E.g., improvement following optimized cardiac therapy")
        sim_sc = st.slider("Simulated Serum Creatinine (mg/dL)", 0.5, 6.0, 1.1, step=0.1, help="E.g., improvement following restored renal perfusion")
        sim_smoking = st.selectbox("Simulated Smoking", ["No", "Yes"], index=0, help="E.g., smoking cessation")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    run_sim = st.button("Run What-If Simulation", use_container_width=True)

    if run_sim:
        try:
            # Baseline patient dict
            curr_dict = {
                "age": float(c_age),
                "anaemia": curr.get("anaemia", 0),
                "creatinine_phosphokinase": curr.get("creatinine_phosphokinase", 582.0),
                "diabetes": 1 if c_diabetes == "Yes" else 0,
                "ejection_fraction": float(c_ef),
                "high_blood_pressure": curr.get("high_blood_pressure", 1),
                "platelets": curr.get("platelets", 263000.0),
                "serum_creatinine": float(c_sc),
                "serum_sodium": float(c_ss),
                "sex": curr.get("sex", 1),
                "smoking": 1 if c_smoking == "Yes" else 0,
            }

            # Simulated patient dict
            sim_dict = dict(curr_dict)
            sim_dict["ejection_fraction"] = float(sim_ef)
            sim_dict["serum_creatinine"] = float(sim_sc)
            sim_dict["smoking"] = 1 if sim_smoking == "Yes" else 0

            # Run existing clinical model twice
            prob_curr, _ = predict_patient_risk(curr_dict)
            prob_sim, _ = predict_patient_risk(sim_dict)

            curr_risk = round(prob_curr * 100.0 if prob_curr <= 1.0 else prob_curr, 2)
            sim_risk = round(prob_sim * 100.0 if prob_sim <= 1.0 else prob_sim, 2)
            risk_change = round(sim_risk - curr_risk, 2)

            st.markdown("<h3 style='margin-top: 1.5rem;'>Simulation Results</h3>", unsafe_allow_html=True)

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown(
                    f"""
                    <div class="tv-card">
                        <div class="tv-card-header">Current Risk</div>
                        <div class="tv-card-value" style="color: {get_risk_color(curr_risk)};">{curr_risk:.2f}%</div>
                        <div class="tv-card-desc">Baseline Clinical Model State</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with scol2:
                st.markdown(
                    f"""
                    <div class="tv-card">
                        <div class="tv-card-header">Simulated Risk</div>
                        <div class="tv-card-value" style="color: {get_risk_color(sim_risk)};">{sim_risk:.2f}%</div>
                        <div class="tv-card-desc">Simulated Parameters State</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with scol3:
                delta_sign = "+" if risk_change > 0 else ""
                delta_color = "#10b981" if risk_change < 0 else ("#ef4444" if risk_change > 0 else "#94a3b8")
                st.markdown(
                    f"""
                    <div class="tv-card">
                        <div class="tv-card-header">Risk Change</div>
                        <div class="tv-card-value" style="color: {delta_color};">{delta_sign}{risk_change:.2f}%</div>
                        <div class="tv-card-desc">Difference in Estimated Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Visual indication
            if risk_change < 0:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                        <strong style="color: #10b981;">✓ Favorable Risk Trajectory:</strong>
                        The simulated parameter modifications indicate an estimated <strong>{abs(risk_change):.2f} percentage point reduction</strong> in predicted risk.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif risk_change > 0:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                        <strong style="color: #ef4444;">⚠ Elevated Risk Trajectory:</strong>
                        The simulated parameter modifications indicate an estimated <strong>+{risk_change:.2f} percentage point increase</strong> in predicted risk.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                        <strong style="color: #94a3b8;">Neutral:</strong> No change in predicted risk between current and simulated states.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Simulation execution error: {str(e)}")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.caption("Note: This is a model-based what-if simulation demonstrating model responsiveness; it does not constitute a clinical intervention.")


# =========================================================
# 5. DIGITAL TWIN TIMELINE
# =========================================================
elif nav_selection == "Digital Twin Timeline":
    st.title("Dynamic Digital Twin Timeline")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem;'>"
        "Show how the patient's state changes across multiple observations."
        "</p>",
        unsafe_allow_html=True,
    )

    # Check if visits exist
    if not st.session_state.patient_history:
        st.markdown(
            """
            <div class="tv-card" style="text-align: center; padding: 2.5rem 1rem;">
                <div style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.6rem;">No patient visits recorded yet.</div>
                <p style="font-size: 0.88rem; color: #64748b; max-width: 500px; margin: 0 auto;">
                    Generate a Patient Digital Twin or run Multimodal Analysis to record clinical encounters.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("Load Representative Longitudinal Demo Encounters"):
            # Load the two representative encounters cited in the research report
            hist = []
            s1 = create_patient_state(
                "Eleanor Vance",
                {"age": 65, "ejection_fraction": 38, "serum_creatinine": 1.2, "serum_sodium": 137, "sex": 0},
                clinical_risk=10.50,
                xray_risk=99.52,
                fused_risk=55.01,
            )
            hist = add_state_to_history(s1, hist)

            s2 = create_patient_state(
                "Eleanor Vance",
                {"age": 65, "ejection_fraction": 28, "serum_creatinine": 1.9, "serum_sodium": 133, "sex": 0},
                clinical_risk=72.00,
                xray_risk=99.52,
                fused_risk=77.51,
            )
            hist = add_state_to_history(s2, hist)

            st.session_state.patient_history = hist
            st.session_state.current_twin = s2
            st.rerun()

    else:
        history = st.session_state.patient_history
        df = history_to_dataframe(history)
        latest = history[-1]

        # Current Digital Twin State
        st.markdown("<h3>Current Digital Twin State</h3>", unsafe_allow_html=True)
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Risk Score</div>
                    <div class="tv-card-value" style="color: {get_risk_color(latest['risk_score'])};">{latest['risk_score']:.2f}%</div>
                    <div class="tv-card-desc">{get_risk_category(latest['risk_score'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tcol2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Ejection Fraction</div>
                    <div class="tv-card-value">{latest['ejection_fraction']:.1f}%</div>
                    <div class="tv-card-desc">Systolic Contractility</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tcol3:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Creatinine</div>
                    <div class="tv-card-value">{latest['serum_creatinine']:.2f} mg/dL</div>
                    <div class="tv-card-desc">Renal Biomarker</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tcol4:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Sodium</div>
                    <div class="tv-card-value">{latest['serum_sodium']:.1f} mEq/L</div>
                    <div class="tv-card-desc">Osmotic Electrolyte</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Patient Risk Trajectory
        st.markdown("<h3>Patient Risk Trajectory</h3>", unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            "Visit": df["Visit"],
            "Risk Score (%)": df["_risk_num"],
        }).set_index("Visit")
        st.line_chart(chart_df, height=280)

        # Clinical State Evolution
        st.markdown("<h3>Clinical State Evolution</h3>", unsafe_allow_html=True)
        param_choice = st.selectbox(
            "Select Clinical Parameter to Inspect:",
            ["Ejection Fraction (%)", "Serum Creatinine (mg/dL)", "Serum Sodium (mEq/L)"],
        )
        col_map = {
            "Ejection Fraction (%)": "_ef_num",
            "Serum Creatinine (mg/dL)": "_sc_num",
            "Serum Sodium (mEq/L)": "_ss_num",
        }
        evol_df = pd.DataFrame({
            "Visit": df["Visit"],
            param_choice: df[col_map[param_choice]],
        }).set_index("Visit")
        st.line_chart(evol_df, height=240)

        # Digital Twin Trend
        st.markdown("<h3>Digital Twin Trend</h3>", unsafe_allow_html=True)
        trend = calculate_risk_trend(history)
        change = get_risk_change(history)

        tr_col1, tr_col2 = st.columns(2)
        with tr_col1:
            sign = "+" if change > 0 else ""
            t_color = "#ef4444" if change > 0 else ("#10b981" if change < 0 else "#94a3b8")
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Overall Risk Change</div>
                    <div class="tv-card-value" style="color: {t_color};">{sign}{change:.2f} percentage points</div>
                    <div class="tv-card-desc">Calculated across latest encounters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tr_col2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Health Trend</div>
                    <div class="tv-card-value">{trend}</div>
                    <div class="tv-card-desc">Longitudinal Directional Classification</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visit History Table
        st.markdown("<h3>Visit History</h3>", unsafe_allow_html=True)
        display_cols = [
            "Visit",
            "Age",
            "Ejection Fraction",
            "Serum Creatinine",
            "Serum Sodium",
            "Clinical Risk",
            "X-Ray Risk",
            "Risk Score",
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# =========================================================
# 6. CLINICAL INTELLIGENCE PAGE
# =========================================================
elif nav_selection == "Clinical Intelligence":
    st.title("Clinical Intelligence")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem;'>"
        "Integrated patient insights from the clinical, imaging, and multimodal prediction pipeline."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.multimodal_result:
        st.info("Run Multimodal Analysis first to generate patient-specific clinical intelligence.")
    else:
        res = st.session_state.multimodal_result
        cd = res.get("clinical_data", {})

        # Overview Risk metrics
        st.markdown("<h3>Multimodal Risk Synthesis</h3>", unsafe_allow_html=True)
        ccol1, ccol2, ccol3, ccol4 = st.columns(4)
        with ccol1:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Clinical Risk</div>
                    <div class="tv-card-value">{res['clinical_risk']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ccol2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">X-Ray Risk</div>
                    <div class="tv-card-value">{res['xray_risk']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ccol3:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Fused Risk</div>
                    <div class="tv-card-value">{res['fused_risk']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ccol4:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Risk Category</div>
                    <div style="margin-top: 0.6rem;">{render_badge(res['category'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Key Clinical Model Factors (Top 5)
        st.markdown("<h3>Key Clinical Model Factors</h3>", unsafe_allow_html=True)
        fi = get_clinical_feature_importance()
        top5 = list(fi.items())[:5]

        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor("#111a2e")
        ax.set_facecolor("#111a2e")

        names = [k.replace("_", " ").title() for k, _ in top5]
        vals = [v * 100 for _, v in top5]

        bars = ax.barh(names[::-1], vals[::-1], color="#00d2c4", edgecolor="none", height=0.6)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.set_xlabel("Relative Decision Importance (%)", color="#94a3b8", fontsize=9)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", color="#f8fafc", va="center", fontsize=8)

        st.pyplot(fig)

        # Current Digital Twin State
        st.markdown("<h3>Current Digital Twin State</h3>", unsafe_allow_html=True)
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Ejection Fraction</div>
                    <div class="tv-card-value">{cd.get('ejection_fraction', 0):.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with scol2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Creatinine</div>
                    <div class="tv-card-value">{cd.get('serum_creatinine', 0):.2f} mg/dL</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with scol3:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Serum Sodium</div>
                    <div class="tv-card-value">{cd.get('serum_sodium', 0):.1f} mEq/L</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Digital Twin Trajectory
        st.markdown("<h3>Digital Twin Trajectory</h3>", unsafe_allow_html=True)
        trend = calculate_risk_trend(st.session_state.patient_history)
        change = get_risk_change(st.session_state.patient_history)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            sign = "+" if change > 0 else ""
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Risk Change</div>
                    <div class="tv-card-value">{sign}{change:.2f} percentage points</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with t_col2:
            st.markdown(
                f"""
                <div class="tv-card">
                    <div class="tv-card-header">Health Trend</div>
                    <div class="tv-card-value">{trend}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # TwinVerse Insight
        st.markdown("<h3>TwinVerse Insight</h3>", unsafe_allow_html=True)
        dominant_factors = core.key_risk_factors(cd, fi, image_abnormal=res.get("is_abnormal", False))
        factors_html = "".join([f"<li>{f}</li>" for f in dominant_factors])

        st.markdown(
            f"""
            <div class="tv-card">
                <div style="color: #cbd5e1; line-height: 1.6; font-size: 0.95rem;">
                    <strong>Model-Based Synthesis:</strong>
                    Based on late fusion, the patient's estimated risk score is <strong>{res['fused_risk']:.2f}%</strong> ({res['category']}).
                    The structured model's decision was predominantly influenced by <em>Serum Creatinine</em> and <em>Ejection Fraction</em>.
                    <br><br>
                    <strong>Identified Contributing Factors:</strong>
                    <ul style="margin: 0.4rem 0 0.8rem 1.2rem; color: #94a3b8;">
                        {factors_html}
                    </ul>
                    <strong>Clinical Action Guideline:</strong><br>
                    {core.recommendation_text(res['category'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
