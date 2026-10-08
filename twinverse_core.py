"""
Core utilities and backwards-compatibility module for TwinVerse.
Bridges legacy utility functions with modern multimodal, explainability, and dynamic state modules.
"""

from models.fusion_model import fuse_predictions, get_risk_category
from models.dynamic_state import (
    create_patient_state,
    add_state_to_history,
    history_to_dataframe,
    calculate_risk_trend,
    get_risk_change,
)
from models.explainability import get_clinical_feature_importance


def fuse_risk(image_risk_pct: float, clinical_risk_pct: float, w_image: float = 0.50, w_clinical: float = 0.50) -> float:
    """Combines imaging and clinical risks into a fused percentage."""
    return fuse_predictions(clinical_risk_pct, image_risk_pct, w_clinical=w_clinical, w_xray=w_image)


def risk_level(pct: float) -> str:
    """Maps risk percentage to category."""
    return get_risk_category(pct)


def build_digital_twin(patient_id, image_risk, clinical_risk, multimodal_risk):
    return {
        "patient_id": patient_id,
        "image_risk": f"{image_risk:.2f}%" if image_risk is not None else "N/A",
        "clinical_risk": f"{clinical_risk:.2f}%",
        "multimodal_risk": f"{multimodal_risk:.2f}%",
        "health_state": f"{get_risk_category(multimodal_risk).upper()}",
    }


def demo_history(current_risk: float):
    v1 = round(max(10.0, current_risk - 44.0), 2)
    v2 = round(max(15.0, current_risk - 35.0), 2)
    v3 = round(max(20.0, current_risk - 23.0), 2)
    return {
        "Visit 1": v1,
        "Visit 2": v2,
        "Visit 3": v3,
        "Current": round(current_risk, 2),
    }


def key_risk_factors(clinical_row: dict, feature_importances: dict = None, image_abnormal: bool = False):
    """
    Identifies dominant patient-specific risk drivers based on clinical thresholds
    and feature importance weights.
    """
    if feature_importances is None:
        feature_importances = get_clinical_feature_importance()

    factors = []
    # Clinical reference thresholds for cardiac & metabolic risk
    ef = clinical_row.get("ejection_fraction", 50)
    sc = clinical_row.get("serum_creatinine", 1.0)
    ss = clinical_row.get("serum_sodium", 137)
    age = clinical_row.get("age", 50)
    cpk = clinical_row.get("creatinine_phosphokinase", 150)
    hbp = clinical_row.get("high_blood_pressure", 0)
    smk = clinical_row.get("smoking", 0)
    dia = clinical_row.get("diabetes", 0)

    if ef <= 35:
        factors.append(f"Severely reduced Ejection Fraction ({ef}%) [Critical Cardiac Risk]")
    elif ef <= 45:
        factors.append(f"Borderline Ejection Fraction ({ef}%)")

    if sc >= 1.5:
        factors.append(f"Elevated Serum Creatinine ({sc} mg/dL) [Impaired Renal Clearance]")

    if ss < 135:
        factors.append(f"Hyponatremia / Low Serum Sodium ({ss} mEq/L)")

    if cpk > 300:
        factors.append(f"Elevated Creatinine Phosphokinase ({cpk} mcg/L)")

    if age >= 65:
        factors.append(f"Advanced age ({int(age)} years)")

    if hbp == 1:
        factors.append("Documented Hypertension (High Blood Pressure)")

    if smk == 1:
        factors.append("Active Tobacco Smoking")

    if dia == 1:
        factors.append("Pre-existing Diabetes Mellitus")

    if image_abnormal:
        factors.append("Cardiomegaly / Cardiac silhouette enlargement on chest radiograph")

    if not factors:
        factors.append("Clinical parameters within conventional physiological ranges")

    return factors


def causal_factor_summary():
    return [
        "Ejection Fraction (Heart pump contractility)",
        "Serum Creatinine (Kidney filtration & metabolic clearance)",
        "Serum Sodium (Fluid balance & neurohormonal activation)",
        "Creatinine Phosphokinase (Myocardial & muscular stress marker)",
        "Age & Blood Pressure (Vascular arterial stiffness)",
        "Cardiomegaly Radiographic Silhouette (Structural cardiac enlargement)",
    ]


def recommendation_text(level: str) -> str:
    if "HIGH" in level.upper():
        return (
            "Comprehensive cardiovascular evaluation and urgent cardiology consultation recommended. "
            "Prioritize neurohormonal therapy optimization, strict fluid balance management, "
            "and serial monitoring of renal biomarkers and ejection fraction."
        )
    elif "MODERATE" in level.upper():
        return (
            "Structured outpatient clinical monitoring recommended. "
            "Focus on aggressive blood pressure control, sodium restriction, lifestyle risk modification, "
            "and repeat echocardiographic/imaging assessment in 3–6 months."
        )
    return (
        "Routine preventive clinical follow-up indicated. Maintain healthy dietary habits, regular physical activity, "
        "and periodic screening of cardiovascular vital signs and renal parameters."
    )
