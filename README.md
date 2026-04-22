![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optuna--tuned-red?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/syedahmadbokhari/breast-cancer-analysis/ci.yml?branch=metabric&label=CI)
![License](https://img.shields.io/badge/License-MIT-green)

# 🧬 Breast Cancer Survival Prediction — METABRIC Dataset

A machine learning project that predicts **breast cancer patient survival outcomes** (Living vs Deceased) using clinical data from the METABRIC dataset.

This project demonstrates a **production-style end-to-end ML workflow**, including data preprocessing, model training, hyperparameter optimisation, explainability, and deployment via API and web interface.

---

## 🔬 Project Overview

This project uses the **METABRIC dataset**, which reflects real-world clinical complexity — unlike simple benchmark datasets.

The system includes:

- Data preprocessing & feature engineering
- Survival analysis with Kaplan-Meier curves
- Training and comparison of multiple ML models
- Hyperparameter optimisation with **Optuna** (40 trials)
- Final model packaged as a **sklearn Pipeline** (scaler + XGBoost)
- Model evaluation using ROC-AUC and classification metrics
- Model explainability (Feature Importance + SHAP)
- REST API using FastAPI
- Interactive UI using Streamlit
- Docker containerisation
- GitHub Actions CI

---

## 📊 Dataset

### METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)

| Property | Value |
|---|---|
| Patients | 2,509 total (~1,981 usable) |
| Target | Overall Survival Status |
| Classes | Living (0) / Deceased (1) |

### Features Used (14 Clinical Features)

- Age at diagnosis
- Tumor size
- Histologic grade
- Lymph node involvement
- Mutation count
- Nottingham Prognostic Index (NPI)
- ER / PR / HER2 receptor status
- Treatment indicators (chemotherapy, radiotherapy, hormone therapy)
- Surgery type
- Menopausal state

> 👉 Focus on **clinical features** improves interpretability and real-world relevance.

---

## 🔧 Machine Learning Pipeline

```text
Data → Preprocessing → Kaplan-Meier EDA → Model Training → Optuna Tuning → Pipeline → Deployment
```

**Key Steps:**

1. Data cleaning & missing value imputation
2. Encoding categorical variables
3. Outlier handling (IQR capping)
4. Kaplan-Meier survival curve analysis
5. Train/test split (70/30, stratified)
6. 6-model comparison (baseline)
7. XGBoost + Optuna hyperparameter tuning (40 trials)
8. Final sklearn Pipeline: StandardScaler → XGBoost
9. Evaluation (ROC, confusion matrix, classification report)
10. Explainability (Feature Importance + SHAP)

---

## 📈 Survival Analysis — Kaplan-Meier Curves

![Kaplan-Meier Curves](./images/kaplan_meier.png)

Survival curves split by ER Status, HER2 Status, and Histologic Grade — showing how clinical subgroups diverge over time.

---

## 🤖 Model Performance

### 📊 Model Comparison

| Model | Accuracy |
|---|---|
| **XGBoost (Optuna-tuned)** | **~75%** |
| SVM | 70.8% |
| Random Forest | 69.6% |
| Logistic Regression | 69.6% |
| KNN | 65.9% |
| Naive Bayes | 64.0% |
| Decision Tree | 60.8% |

**Final Model: XGBoost + StandardScaler Pipeline**
- Tuned with Optuna (40 trials, 5-fold CV ROC-AUC objective)
- `scale_pos_weight` for class imbalance
- Packaged as a single `pipeline.pkl` — no separate scaler needed

---

### 📈 ROC Curve

![ROC Curve](./images/roc_curve.png)

**ROC-AUC: ~0.79** after Optuna tuning (vs 0.73 baseline Random Forest)

---

### 🔍 Feature Importance

![Feature Importance](./images/feature_importance.png)

**Top contributing features:**

1. Nottingham Prognostic Index
2. Tumor Size
3. Age at Diagnosis
4. Lymph Node Involvement

> 👉 These align with established clinical risk factors for breast cancer survival.

---

## 🚀 Deployment

### 🧠 Architecture

```
User → Streamlit UI → FastAPI → sklearn Pipeline → Prediction
```

### 🖥 Streamlit App

```bash
streamlit run notebooks/app.py
```

- Interactive UI with sidebar inputs grouped by clinical section
- Colour-coded result card (green = Living, red = Deceased)
- Probability bars with confidence score

### ⚡ FastAPI

```bash
uvicorn notebooks.api:app --reload
```

Interactive docs at `http://localhost:8000/docs`

### 🧪 Example Request

```json
{
  "age_at_diagnosis": 65.0,
  "tumor_size": 28.0,
  "neoplasm_histologic_grade": 2.0,
  "lymph_nodes_examined_positive": 1.0,
  "mutation_count": 3.0,
  "nottingham_prognostic_index": 4.5,
  "er_status": 1,
  "her2_status": 0,
  "pr_status": 1,
  "chemotherapy": 0,
  "hormone_therapy": 1,
  "radio_therapy": 1,
  "type_of_breast_surgery": 0,
  "inferred_menopausal_state": 1
}
```

**Example Response:**

```json
{
  "prediction": "Living",
  "living_probability": 0.71,
  "deceased_probability": 0.29,
  "confidence": 71.0
}
```

### 📦 Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

---

## 📂 Project Structure

```
Breast-Cancer-ML/
│
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI
├── data/
│   └── Breast Cancer METABRIC.csv
├── models/
│   ├── pipeline.pkl                     # XGBoost Pipeline (scaler + model)
│   ├── breast_cancer_model.pkl          # Baseline Random Forest (reference)
│   └── scaler.pkl
├── notebooks/
│   ├── Breast-Cancer-Analysis.ipynb     # Full ML pipeline
│   ├── app.py                           # Streamlit UI
│   ├── api.py                           # FastAPI REST endpoint
│   └── predict.py                       # Standalone script
├── images/
│   ├── kaplan_meier.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── MODEL_CARD.md                        # Model documentation
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- Moderate accuracy (~75%) due to real-world data complexity
- Clinical features only — no genomic or gene expression data
- No time-to-event survival modelling (classification only)
- Single cohort (UK/Canada) — may not generalise globally

---

## 🔮 Future Improvements

- Add Cox proportional hazards survival model
- Incorporate gene expression features
- Implement request logging and model monitoring
- Add API authentication
- Deploy with Kubernetes / AWS ECS

---

## 🧠 Key Takeaways

- Handling real-world healthcare datasets
- Building production-style ML pipelines with sklearn `Pipeline`
- Hyperparameter optimisation with Optuna
- Deploying ML models via API + UI
- Interpreting predictions responsibly with SHAP

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not intended for medical use. See [MODEL_CARD.md](./MODEL_CARD.md) for full limitations and ethical considerations.
