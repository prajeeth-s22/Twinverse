"""
Multimodal Fusion Model for TwinVerse.
Combines structured clinical risk prediction and chest X-ray Cardiomegaly estimation
using late probability-level fusion to produce an integrated patient risk score.
"""


def fuse_predictions(clinical_risk: float, xray_risk: float, w_clinical: float = 0.50, w_xray: float = 0.50) -> float:
    """
    Combines clinical risk and chest X-ray risk using weighted late fusion.

    Parameters:
        clinical_risk: Risk score from clinical model (0-1 or 0-100)
        xray_risk: Risk score from chest X-ray model (0-1 or 0-100)
        w_clinical: Weight for clinical branch (default 0.50)
        w_xray: Weight for imaging branch (default 0.50)

    Returns:
        fused_risk_pct (float): Combined risk percentage (0.0 to 100.0) rounded to 2 decimals.
    """
    # Normalize inputs to 0-100 percentage scale
    c_pct = clinical_risk * 100.0 if (0.0 <= clinical_risk <= 1.0) else float(clinical_risk)
    x_pct = xray_risk * 100.0 if (0.0 <= xray_risk <= 1.0) else float(xray_risk)

    fused_pct = (w_clinical * c_pct) + (w_xray * x_pct)
    return round(float(fused_pct), 2)


def get_risk_category(risk_score: float) -> str:
    """
    Maps a risk score into standard clinical risk tiers:
    - Low Risk (< 40%)
    - Moderate Risk (40% - 69.9%)
    - High Risk (>= 70%)

    Parameters:
        risk_score: Risk value (either 0-100 percentage or 0-1 probability)

    Returns:
        category (str): 'Low Risk', 'Moderate Risk', or 'High Risk'
    """
    pct = risk_score * 100.0 if (0.0 <= risk_score <= 1.0) else float(risk_score)

    if pct >= 70.0:
        return "High Risk"
    elif pct >= 40.0:
        return "Moderate Risk"
    else:
        return "Low Risk"
