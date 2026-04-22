# Model Card — Breast Cancer Survival Predictor

## Model Details

| Property | Value |
|---|---|
| **Model type** | XGBoost Classifier (sklearn Pipeline) |
| **Tuning** | Optuna — 40 trials, maximising 5-fold CV ROC-AUC |
| **Pipeline steps** | StandardScaler → XGBClassifier |
| **Version** | 2.0 |
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
| **Accuracy** | ~75% |
| **ROC-AUC** | ~0.79 |
| **CV ROC-AUC (5-fold)** | ~0.78 |

Evaluated on held-out 30% test set (stratified split, random_state=40).

## Limitations

- **Clinical features only** — no genomic or gene expression data, which are strong predictors
- **Classification only** — does not model time-to-event (no survival curves per patient)
- **Single cohort** — trained on METABRIC; may not generalise to other populations or treatment eras
- **Missing data** — rows with missing survival status dropped (~21%); imputation used for features
- **Class imbalance** — more deceased than living patients; handled via `scale_pos_weight`

## Bias and Fairness Considerations

- Dataset is predominantly female (consistent with breast cancer epidemiology)
- Age and menopausal state are correlated — model may reflect demographic patterns rather than causal factors
- METABRIC is a UK/Canada cohort — outcomes may differ for other ethnicities or healthcare systems
- Treatment indicators (chemo, hormone, radio) reflect past clinical decisions, not purely biological factors

## Ethical Considerations

This model **must not** be used to make or influence clinical decisions about real patients. Survival prediction is inherently uncertain and involves factors beyond the scope of this dataset. Any clinical use requires regulatory approval, prospective validation, and clinician oversight.

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
