"""
Generates a synthetic clinical dataset for TwinVerse (Review-2 prototype).

IMPORTANT (documented per project guidance):
This is NOT a real patient dataset. It is a procedurally generated dataset whose
feature -> risk relationships are built to mimic well-known, textbook clinical
associations (age, blood pressure, glucose, BMI -> cardiometabolic risk), so
that the trained model behaves sensibly for demo purposes. It should be
described in the demo as a prototype trained on synthetic data, not a
clinically validated model.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
N = 4000

age = RNG.normal(52, 15, N).clip(18, 90)
bp = RNG.normal(128, 18, N).clip(85, 210)
glucose = RNG.normal(110, 35, N).clip(60, 300)
bmi = RNG.normal(26, 5, N).clip(15, 50)

# Plausible clinical risk score (0-1) built from normalized weighted factors
def norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)

risk_score = (
    0.28 * norm(glucose, 70, 220) +
    0.24 * norm(bp, 100, 190) +
    0.20 * norm(age, 20, 85) +
    0.16 * norm(bmi, 18, 42) +
    0.12 * RNG.normal(0, 1, N).clip(-1, 1) * 0.5  # noise
)
risk_score = np.clip(risk_score, 0, 1)

# Threshold chosen (median-ish) to yield a reasonably balanced demo dataset
label = (risk_score > np.quantile(risk_score, 0.55)).astype(int)  # 1 = HIGH RISK

df = pd.DataFrame({
    "age": age.round(1),
    "blood_pressure": bp.round(1),
    "glucose": glucose.round(1),
    "bmi": bmi.round(1),
    "risk_label": label,
})

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "clinical_dataset.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["risk_label"].value_counts(normalize=True))
