"""
Dynamic Digital Twin State Management for TwinVerse.
Maintains longitudinal patient health states, trajectories, and trend analysis.
"""

import datetime
import pandas as pd


def create_patient_state(
    patient_name: str,
    patient_data: dict,
    clinical_risk: float,
    xray_risk: float = None,
    fused_risk: float = None,
    visit_number: int = None,
    timestamp: str = None,
) -> dict:
    """
    Creates a structured dictionary representing the patient's state at a single encounter.
    """
    # Normalize risks to percentages
    c_pct = round(clinical_risk * 100.0 if (0.0 <= clinical_risk <= 1.0) else float(clinical_risk), 2)
    x_pct = (
        round(xray_risk * 100.0 if (0.0 <= xray_risk <= 1.0) else float(xray_risk), 2)
        if xray_risk is not None
        else None
    )
    f_pct = (
        round(fused_risk * 100.0 if (0.0 <= fused_risk <= 1.0) else float(fused_risk), 2)
        if fused_risk is not None
        else c_pct
    )

    now_str = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Sex decode
    raw_sex = patient_data.get("sex", 1)
    sex_label = "Male" if raw_sex in [1, "1", "Male", "male", "M"] else "Female"

    state = {
        "visit": visit_number or 1,
        "timestamp": now_str,
        "name": patient_name or "Anonymous Patient",
        "age": float(patient_data.get("age", 60)),
        "sex": sex_label,
        "ejection_fraction": float(patient_data.get("ejection_fraction", 38)),
        "serum_creatinine": float(patient_data.get("serum_creatinine", 1.1)),
        "serum_sodium": float(patient_data.get("serum_sodium", 136)),
        "clinical_risk": c_pct,
        "xray_risk": x_pct,
        "fused_risk": f_pct,
        "risk_score": f_pct,
        "clinical_data": dict(patient_data),
    }
    return state


def add_state_to_history(state: dict, history: list = None) -> list:
    """
    Appends a new patient state to history list, updating visit sequence number.
    """
    if history is None:
        history = []

    state["visit"] = len(history) + 1
    history.append(state)
    return history


def history_to_dataframe(history: list) -> pd.DataFrame:
    """
    Transforms a list of patient visit states into a clean pandas DataFrame.
    Columns include: Visit, Age, Ejection Fraction, Serum Creatinine,
    Serum Sodium, Clinical Risk, X-Ray Risk, Risk Score.
    """
    if not history:
        return pd.DataFrame(
            columns=[
                "Visit",
                "Age",
                "Ejection Fraction",
                "Serum Creatinine",
                "Serum Sodium",
                "Clinical Risk",
                "X-Ray Risk",
                "Risk Score",
            ]
        )

    rows = []
    for item in history:
        rows.append({
            "Visit": f"Visit {item.get('visit', 1)}",
            "Age": int(item.get("age", 0)),
            "Ejection Fraction": float(item.get("ejection_fraction", 0.0)),
            "Serum Creatinine": float(item.get("serum_creatinine", 0.0)),
            "Serum Sodium": float(item.get("serum_sodium", 0.0)),
            "Clinical Risk": f"{item.get('clinical_risk', 0.0):.2f}%",
            "X-Ray Risk": f"{item.get('xray_risk', 0.0):.2f}%" if item.get("xray_risk") is not None else "N/A",
            "Risk Score": f"{item.get('risk_score', 0.0):.2f}%",
            "_risk_num": float(item.get("risk_score", 0.0)),
            "_ef_num": float(item.get("ejection_fraction", 0.0)),
            "_sc_num": float(item.get("serum_creatinine", 0.0)),
            "_ss_num": float(item.get("serum_sodium", 0.0)),
        })
    return pd.DataFrame(rows)


def calculate_risk_trend(history: list) -> str:
    """
    Determines patient health trajectory across visits:
    - 'Increasing' (risk rising)
    - 'Decreasing' (risk improving)
    - 'Stable'
    """
    if not history or len(history) < 2:
        return "Stable"

    latest_risk = float(history[-1].get("risk_score", 0.0))
    prev_risk = float(history[-2].get("risk_score", 0.0))
    diff = latest_risk - prev_risk

    if diff > 1.5:
        return "Increasing"
    elif diff < -1.5:
        return "Decreasing"
    else:
        return "Stable"


def get_risk_change(history: list) -> float:
    """
    Calculates numerical change in risk percentage points between latest and previous visits.
    """
    if not history or len(history) < 2:
        return 0.0

    latest_risk = float(history[-1].get("risk_score", 0.0))
    prev_risk = float(history[-2].get("risk_score", 0.0))
    return round(latest_risk - prev_risk, 2)
