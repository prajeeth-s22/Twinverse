"""
Clinical Prediction Model for TwinVerse.
Predicts patient cardiovascular risk from 11 structured clinical features
using a trained Random Forest Classifier on the Heart Failure Clinical Records dataset.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "clinical_model.joblib")
METRICS_PATH = os.path.join(HERE, "clinical_metrics.json")
DATA_PATH = os.path.join(HERE, "..", "data", "heart_failure_clinical_records.csv")

CLINICAL_FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]

_model_cache = None


def load_clinical_model():
    """Loads the trained clinical model and feature list."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache["model"], _model_cache["features"]

    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        _model_cache = bundle
        return bundle["model"], bundle.get("features", CLINICAL_FEATURES)

    # Fallback: train on the dataset if model file not yet dumped
    if os.path.exists(DATA_PATH):
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_csv(DATA_PATH)
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        )
        clf.fit(df[CLINICAL_FEATURES], df["DEATH_EVENT"])
        _model_cache = {"model": clf, "features": CLINICAL_FEATURES}
        try:
            joblib.dump(_model_cache, MODEL_PATH)
        except Exception:
            pass
        return clf, CLINICAL_FEATURES

    raise FileNotFoundError(f"Neither model file {MODEL_PATH} nor dataset {DATA_PATH} was found.")


def predict_patient_risk(patient_data: dict):
    """
    Predicts patient cardiovascular risk from clinical features.

    Parameters:
        patient_data (dict): Dictionary with 11 clinical features:
            - age (float/int)
            - anaemia (0 or 1)
            - creatinine_phosphokinase (float/int)
            - diabetes (0 or 1)
            - ejection_fraction (float/int)
            - high_blood_pressure (0 or 1)
            - platelets (float/int)
            - serum_creatinine (float/int)
            - serum_sodium (float/int)
            - sex (0 for Female, 1 for Male)
            - smoking (0 or 1)

    Returns:
        risk_probability (float): Risk probability between 0.0 and 1.0
        prediction (int): 1 if High Risk / Event predicted, 0 otherwise
    """
    model, features = load_clinical_model()

    # Build input row in exact order of features
    row_values = []
    for f in features:
        val = patient_data.get(f)
        if val is None:
            # Defaults based on median population baselines
            defaults = {
                "age": 60.0,
                "anaemia": 0,
                "creatinine_phosphokinase": 250.0,
                "diabetes": 0,
                "ejection_fraction": 38.0,
                "high_blood_pressure": 0,
                "platelets": 250000.0,
                "serum_creatinine": 1.1,
                "serum_sodium": 136.0,
                "sex": 1,
                "smoking": 0,
            }
            val = defaults.get(f, 0.0)
        row_values.append(float(val))

    df_row = pd.DataFrame([row_values], columns=features)
    proba = model.predict_proba(df_row)[0]

    # Target class 1 is event / high risk
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    risk_prob = float(proba[pos_idx])
    pred = int(model.predict(df_row)[0])

    return risk_prob, pred
