# Model Card — Breast Cancer Survival Predictor

## Model Details

| Property | Value |
|---|---|
| **Model type** | XGBoost Classifier (sklearn Pipeline) |
| **Tuning** | Optuna — 40 trials, maximising 5-fold CV ROC-AUC |
| **Pipeline steps** | StandardScaler → XGBClassifier |
| **Version** | 3.0 |
| **Artifact** | `models/pipeline.pkl` |
| **Training date** | April 2026 |

## Intended Use

- **Primary use:** Educational and research exploration of clinical survival prediction
- **Intended users:** Data scientists, ML researchers, students
- **Out-of-scope uses:** Clinical decision making, real patient diagnosis, treatment planning

## Dataset

| Property | Value |
|---|---|
| **Name** | METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) |
| **Patients** | 2,509 total — 1,981 usable after removing missing targets |
| **Split** | 70% train / 30% test, stratified by target |
| **Target** | Overall Survival Status — Deceased (1) / Living (0) |
| **Class balance** | 837 Living / 1,144 Deceased |

## Features

| Feature | Type | Clinical Meaning |
|---|---|---|
| `age_at_diagnosis` | Numeric | Patient age at time of diagnosis |
| `tumor_size` | Numeric | Tumour size in mm |
| `neoplasm_histologic_grade` | Numeric (1–3) | Tumour aggressiveness grade |
| `lymph_nodes_examined_positive` | Numeric | Number of positive lymph nodes |
| `mutation_count` | Numeric | Total somatic mutation burden |
| `nottingham_prognostic_index` | Numeric | Composite clinical severity score |
| `er_status` | Binary | Estrogen receptor positive/negative |
| `her2_status` | Binary | HER2 receptor positive/negative |
| `pr_status` | Binary | Progesterone receptor positive/negative |
| `chemotherapy` | Binary | Received chemotherapy |
| `hormone_therapy` | Binary | Received hormone therapy |
| `radio_therapy` | Binary | Received radiotherapy |
| `type_of_breast_surgery` | Binary | Mastectomy vs breast conserving |
| `inferred_menopausal_state` | Binary | Post vs pre-menopausal |

## Performance

| Metric | Value |
|---|---|
| **Accuracy** | ~69% |
| **ROC-AUC** | 0.74 |
| **5-fold CV ROC-AUC** | mean ± std reported in notebook |
| **Cox PH C-index** | reported in notebook |

Evaluated on held-out 30% test set (stratified split, random_state=40).
Fully reproducible: Optuna `TPESampler(seed=42)` + `XGBClassifier(random_state=42)`.

## Survival Modelling (Cox PH)

A Cox Proportional Hazards model was fitted alongside the classifier to model **time-to-event** data using `Overall Survival (Months)`. Outputs include hazard ratios with 95% CIs for all 14 features, predicted survival curves for high-risk vs low-risk archetypes, and a concordance index (C-index) for direct comparison to XGBoost ROC-AUC.

## Robust Validation

Stratified 5-fold cross-validation reports mean ± std for ROC-AUC, accuracy, and F1. Learning curves diagnose underfitting vs overfitting across training set sizes.

## Error Analysis

False negatives (Deceased predicted as Living) are the highest-risk errors. The analysis compares mean feature profiles of false negatives vs true positives, identifying where the model is most likely to miss high-risk patients.

## Probability Calibration

Raw XGBoost probabilities are calibrated with Platt scaling (`CalibratedClassifierCV`, 3-fold). The calibration curve demonstrates before/after improvement, ensuring predicted probabilities reflect true empirical risk.

## Risk Levels (API & UI)

| Deceased Probability | Risk Level |
|---|---|
| ≥ 70% | High |
| 50–70% | Moderate |
| < 50% | Lower |

## Prediction Monitoring

Every API call is logged to `logs/predictions.csv` with timestamp, request ID, prediction, probabilities, confidence, and all input features. The `/logs` endpoint returns the last N records.

## Limitations

- **Clinical features only** — no genomic or gene expression data
- **Classification only** — deployed model does not produce time-to-event curves per patient
- **Single cohort** — METABRIC is UK/Canada; may not generalise globally
- **Missing data** — ~21% of rows dropped due to missing survival status
- **Class imbalance** — handled via `scale_pos_weight`

## Bias and Fairness Considerations

- Dataset is predominantly female
- Age and menopausal state are correlated — model may capture demographic patterns
- METABRIC is a UK/Canada cohort — outcomes may differ across ethnicities and healthcare systems
- Treatment indicators reflect past clinical decisions, not purely biological factors

## Ethical Considerations

This model **must not** be used to make or influence clinical decisions about real patients. Any clinical use requires regulatory approval, prospective validation, and clinician oversight.

## How to Use

```python
import joblib, pandas as pd

pipeline = joblib.load("models/pipeline.pkl")

sample = pd.DataFrame([[65, 28, 2, 1, 3, 4.5, 1, 0, 1, 0, 1, 1, 0, 1]], columns=[
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
])

prediction  = pipeline.predict(sample)          # 0=Living, 1=Deceased
probability = pipeline.predict_proba(sample)    # [P(Living), P(Deceased)]
```

## Citation

> Pereira, B. et al. (2016). The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. *Nature Communications*, 7, 11479.
