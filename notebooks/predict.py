import joblib
import numpy as np
import pandas as pd

model  = joblib.load("models/breast_cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
]

# Sample patient: 65-year-old, 28mm tumor, grade 2, 1 positive lymph node,
# 3 mutations, NPI=4.5, ER+, HER2-, PR+, no chemo, hormone therapy, radiotherapy,
# breast conserving surgery, post-menopausal
sample = pd.DataFrame([[
    65.0,   # age_at_diagnosis
    28.0,   # tumor_size (mm)
    2.0,    # neoplasm_histologic_grade (1/2/3)
    1.0,    # lymph_nodes_examined_positive
    3.0,    # mutation_count
    4.5,    # nottingham_prognostic_index
    1.0,    # er_status (1=Positive)
    0.0,    # her2_status (0=Negative)
    1.0,    # pr_status (1=Positive)
    0.0,    # chemotherapy (0=No)
    1.0,    # hormone_therapy (1=Yes)
    1.0,    # radio_therapy (1=Yes)
    0.0,    # type_of_breast_surgery (0=Breast Conserving)
    1.0,    # inferred_menopausal_state (1=Post)
]], columns=FEATURE_NAMES)

sample_scaled = scaler.transform(sample)
prediction    = model.predict(sample_scaled)
probability   = model.predict_proba(sample_scaled)

print("Raw prediction:", prediction[0])
print("Predicted:", "Deceased" if prediction[0] == 1 else "Living")
print(f"Confidence: {np.max(probability) * 100:.2f}%")
print(f"Living probability:   {probability[0][0]:.4f}")
print(f"Deceased probability: {probability[0][1]:.4f}")
