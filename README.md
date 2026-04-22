# Breast Cancer Survival Prediction — METABRIC Dataset

A machine learning project that predicts breast cancer patient survival outcome (Living or Deceased) using clinical data from the METABRIC dataset. Includes a trained Random Forest model, a Streamlit web app, and a FastAPI REST endpoint.

---

## Dataset

**METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium)

| Property | Value |
|---|---|
| Source | Kaggle — Breast Cancer Gene Expression Profiles (METABRIC) |
| Patients | 2,509 total — 1,981 usable after removing missing targets |
| Target | `Overall Survival Status` (Living = 0, Deceased = 1) |
| Class split | 837 Living / 1,144 Deceased |

### Features Used (14)

| Feature | Type | Description |
|---|---|---|
| `age_at_diagnosis` | Numeric | Patient age at diagnosis |
| `tumor_size` | Numeric | Tumor size in mm |
| `neoplasm_histologic_grade` | Numeric | Grade 1 / 2 / 3 |
| `lymph_nodes_examined_positive` | Numeric | Number of positive lymph nodes |
| `mutation_count` | Numeric | Total somatic mutation count |
| `nottingham_prognostic_index` | Numeric | Composite clinical severity score |
| `er_status` | Encoded | Estrogen receptor status (1=Positive, 0=Negative) |
| `her2_status` | Encoded | HER2 receptor status (1=Positive, 0=Negative) |
| `pr_status` | Encoded | Progesterone receptor status (1=Positive, 0=Negative) |
| `chemotherapy` | Encoded | Received chemotherapy (1=Yes, 0=No) |
| `hormone_therapy` | Encoded | Received hormone therapy (1=Yes, 0=No) |
| `radio_therapy` | Encoded | Received radiotherapy (1=Yes, 0=No) |
| `type_of_breast_surgery` | Encoded | Surgery type (1=Mastectomy, 0=Breast Conserving) |
| `inferred_menopausal_state` | Encoded | Menopausal state (1=Post, 0=Pre) |

---

## Project Structure

```
Breast-Cancer-ML/
├── data/
│   └── Breast Cancer METABRIC.csv   # METABRIC clinical dataset
├── notebooks/
│   ├── Breast-Cancer-Analysis.ipynb # Full ML pipeline
│   ├── app.py                       # Streamlit prediction web app
│   ├── api.py                       # FastAPI REST endpoint
│   └── predict.py                   # Standalone prediction script
├── models/
│   ├── breast_cancer_model.pkl      # Trained Random Forest model
│   └── scaler.pkl                   # Fitted StandardScaler
├── images/
│   ├── feature_importance.png
│   └── roc_curve.png
├── requirements.txt
└── Dockerfile
```

---

## ML Pipeline (Notebook)

1. **Load & explore** METABRIC CSV
2. **Preprocess** — encode categorical variables, impute missing values with median/mode
3. **EDA** — survival distributions, feature histograms, correlation heatmap
4. **Outlier handling** — IQR capping on numeric features
5. **Clustering** — K-Means (k=2) with PCA projection
6. **Train/test split** — 70/30, stratified
7. **Scale** — StandardScaler fitted on training set
8. **Train & compare** 6 classifiers
9. **Evaluate** — classification report, confusion matrix, ROC curve
10. **Explain** — Random Forest feature importance + SHAP beeswarm

### Model Comparison

| Model | Accuracy |
|---|---|
| SVM | 70.8% |
| Random Forest | 69.6% |
| Logistic Regression | 69.6% |
| KNN | 65.9% |
| Naive Bayes | 64.0% |
| Decision Tree | 60.8% |

**Saved model:** Random Forest — `n_estimators=200`, `class_weight='balanced'`
**ROC-AUC:** 0.73

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Streamlit Web App

```bash
streamlit run notebooks/app.py
```

Opens a browser UI where you enter patient clinical data and get an instant survival prediction with confidence score.

### FastAPI

```bash
uvicorn notebooks.api:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_at_diagnosis": 65.0,
    "tumor_size": 28.0,
    "neoplasm_histologic_grade": 2.0,
    "lymph_nodes_examined_positive": 1.0,
    "mutation_count": 3.0,
    "nottingham_prognostic_index": 4.5,
    "er_status": 1.0,
    "her2_status": 0.0,
    "pr_status": 1.0,
    "chemotherapy": 0.0,
    "hormone_therapy": 1.0,
    "radio_therapy": 1.0,
    "type_of_breast_surgery": 0.0,
    "inferred_menopausal_state": 1.0
  }'
```

**Example response:**

```json
{
  "prediction": "Living",
  "living_probability": 0.6300,
  "deceased_probability": 0.3700
}
```

### Standalone Script

```bash
python notebooks/predict.py
```

### Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

---

## Feature Encoding Reference

| Feature | Value | Encoded |
|---|---|---|
| ER / HER2 / PR Status | Positive | 1 |
| ER / HER2 / PR Status | Negative | 0 |
| Chemotherapy / Hormone Therapy / Radio Therapy | Yes | 1 |
| Chemotherapy / Hormone Therapy / Radio Therapy | No | 0 |
| Type of Breast Surgery | Mastectomy | 1 |
| Type of Breast Surgery | Breast Conserving | 0 |
| Inferred Menopausal State | Post | 1 |
| Inferred Menopausal State | Pre | 0 |
| Overall Survival Status (target) | Deceased | 1 |
| Overall Survival Status (target) | Living | 0 |

---

## Disclaimer

This project is for educational and research purposes only. Predictions made by this model are **not** a substitute for professional medical diagnosis or clinical judgement.
