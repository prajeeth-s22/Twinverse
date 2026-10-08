"""
Trains the clinical risk-prediction model for TwinVerse.
Model: RandomForestClassifier (simple, interpretable, fast — per project guidance
to start with Random Forest / XGBoost rather than deep learning for Review-2).
"""
import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "clinical_dataset.csv")
MODEL_PATH = os.path.join(HERE, "clinical_model.joblib")
METRICS_PATH = os.path.join(HERE, "clinical_metrics.json")

FEATURES = ["age", "blood_pressure", "glucose", "bmi"]
TARGET = "risk_label"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": cm,
        "features": FEATURES,
        "feature_importances": {
            f: round(float(imp), 4) for f, imp in zip(FEATURES, clf.feature_importances_)
        },
    }

    joblib.dump({"model": clf, "features": FEATURES}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", MODEL_PATH)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
