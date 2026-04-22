# 🧬 Breast Cancer Survival Prediction — METABRIC Dataset

A machine learning project that predicts **breast cancer patient survival outcomes** (Living vs Deceased) using clinical data from the METABRIC dataset.

This project demonstrates a **production-style end-to-end ML workflow**, including data preprocessing, model training, explainability, and deployment via API and web interface.

---

## 🔬 Project Overview

This project upgrades from a simple benchmark dataset to the **METABRIC dataset**, which reflects real-world clinical complexity.

The system includes:

- Data preprocessing & feature engineering
- Training and comparison of multiple ML models
- Model evaluation using ROC-AUC and classification metrics
- Model explainability (Feature Importance + SHAP)
- REST API using FastAPI
- Interactive UI using Streamlit
- Docker containerization

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
Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

**Key Steps:**

1. Data cleaning & missing value imputation
2. Encoding categorical variables
3. Outlier handling (IQR capping)
4. Feature scaling (StandardScaler)
5. Train/test split (70/30, stratified)
6. Model training (6 algorithms)
7. Evaluation (ROC, confusion matrix)
8. Explainability (Feature Importance + SHAP)

---

## 🤖 Model Performance

### 📊 Model Comparison

| Model | Accuracy |
|---|---|
| SVM | 70.8% |
| Random Forest | 69.6% |
| Logistic Regression | 69.6% |
| KNN | 65.9% |
| Naive Bayes | 64.0% |
| Decision Tree | 60.8% |

**Selected Model: Random Forest**
- `n_estimators = 200`
- `class_weight = 'balanced'`

---

### 📈 ROC Curve

![ROC Curve](./images/roc_curve.png)

- **ROC-AUC: 0.73**
- Demonstrates realistic performance on clinical data

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
User → Streamlit UI → FastAPI → ML Model → Prediction
```

### 🖥 Streamlit App

```bash
streamlit run notebooks/app.py
```

- Interactive UI for entering patient data
- Real-time prediction with probabilities

### ⚡ FastAPI

```bash
uvicorn notebooks.api:app --reload
```

API available at `http://localhost:8000/docs`

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
├── data/
│   └── Breast Cancer METABRIC.csv
├── models/
│   ├── breast_cancer_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── Breast-Cancer-Analysis.ipynb
│   ├── app.py
│   ├── api.py
│   └── predict.py
├── images/
│   ├── feature_importance.png
│   └── roc_curve.png
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- Moderate accuracy (~70%) due to real-world data complexity
- Only clinical features used (no genomic data)
- No time-to-event survival modeling (classification only)

---

## 🔮 Future Improvements

- Add survival analysis (Cox proportional hazards model)
- Incorporate gene expression features
- Implement monitoring & logging
- Add API authentication
- Deploy using Kubernetes / AWS ECS

---

## 🧠 Key Takeaways

- Handling real-world healthcare datasets
- Building production-style ML pipelines
- Deploying ML models via API + UI
- Interpreting predictions responsibly

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not intended for medical use.
