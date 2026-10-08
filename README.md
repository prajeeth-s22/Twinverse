# TwinVerse — A Human Digital Twin Framework for Personalized Disease Prediction

**TwinVerse** is an interactive multimodal medical AI application that constructs a dynamic digital twin of a patient by combining structured clinical biomarkers and chest radiograph analysis.

---

## 🏛️ System Architecture

```
                 PATIENT
                    |
          ----------------------
          |                    |
   Clinical Data          Chest X-Ray
          |                    |
   Clinical Model           ResNet18 / CNN
          |                    |
   Clinical Risk        Cardiomegaly Risk
          |                    |
          ----------- -----------
                     |
                Late Fusion
                     |
              Fused Risk Score
                     |
        -------------------------
        |           |           |
   Explainability  Timeline  What-If
        |           |           |
    Grad-CAM     Digital Twin  Simulation
                     |
             Clinical Intelligence
```

---

## 📦 Project Structure

```
TwinVerse/
├── app.py                     # Main Streamlit application (6 pages)
├── twinverse_core.py          # Core logic & backward compatibility layer
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── models/
│   ├── clinical_model.py      # predict_patient_risk() (11 clinical features)
│   ├── clinical_model.joblib  # Trained Random Forest classifier
│   ├── clinical_metrics.json  # Quantitative evaluation metrics
│   ├── train_clinical.py      # Script to train clinical model
│   ├── xray_inference.py      # predict_xray() (Cardiomegaly estimation)
│   ├── xray_cnn.py            # Convolutional neural network architecture
│   ├── xray_model.pt          # Trained X-ray model checkpoint
│   ├── resnet18_cardiomegaly.pt # Trained ResNet18 checkpoint
│   ├── xray_metrics.json      # Evaluation metrics
│   ├── train_xray.py          # Script to train X-ray model
│   ├── explainability.py      # generate_gradcam() & get_clinical_feature_importance()
│   ├── gradcam.py             # Gradient-weighted Class Activation Mapping
│   ├── fusion_model.py        # fuse_predictions() & get_risk_category()
│   └── dynamic_state.py       # Digital twin state tracking & trajectory
├── data/
│   ├── heart_failure_clinical_records.csv  # 11-feature clinical dataset
│   ├── clinical_dataset.csv
│   ├── make_clinical_dataset.py
│   ├── make_xray_dataset.py
│   └── xray_synth/            # Synthesized chest radiographs (normal & abnormal)
└── assets/
    └── demo_patient_xray.png  # Sample radiograph for demonstrations
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` (or the port specified by Streamlit).

---

## 🔬 Core Components

1. **Clinical Prediction Model** (`models/clinical_model.py`):
   - 11 Clinical Features: `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`.
   - Function: `predict_patient_risk(patient_data)` returns `(risk_probability, prediction)`.
2. **Chest X-Ray Analysis** (`models/xray_inference.py`):
   - Predicts Cardiomegaly probability from uploaded radiographs.
   - Function: `predict_xray(image_input)`.
3. **Multimodal Late Fusion** (`models/fusion_model.py`):
   - Probability-level late fusion combining imaging and laboratory biomarker branches.
   - Functions: `fuse_predictions()`, `get_risk_category()`.
4. **Explainable AI** (`models/explainability.py`):
   - Dual-modality transparency: Grad-CAM attention heatmaps and Random Forest feature importance rankings.
5. **Dynamic Digital Twin Timeline** (`models/dynamic_state.py`):
   - Tracks longitudinal patient encounters, computes risk trajectories, and classifies health trends (`Increasing`, `Decreasing`, `Stable`).
6. **What-If Health Simulation**:
   - Model-based sensitivity analysis exploring biomarker modifications without causal overclaims.
7. **Clinical Intelligence Dashboard**:
   - Unified multi-perspective synthesis of patient risk, decision drivers, and longitudinal history.
