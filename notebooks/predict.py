import joblib
import pandas as pd

pipeline = joblib.load("models/pipeline.pkl")

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
]

# Sample: 65-year-old, 28mm tumor, grade 2, 1 positive lymph node,
# 3 mutations, NPI=4.5, ER+, HER2-, PR+, no chemo, hormone+radio therapy,
# breast conserving surgery, post-menopausal
sample = pd.DataFrame([[
    65.0, 28.0, 2.0, 1.0, 3.0, 4.5,
    1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0
]], columns=FEATURE_NAMES)

prediction  = pipeline.predict(sample)
probability = pipeline.predict_proba(sample)

print("Predicted:", "Deceased" if prediction[0] == 1 else "Living")
print(f"Confidence:          {max(probability[0]) * 100:.2f}%")
print(f"Living probability:  {probability[0][0]:.4f}")
print(f"Deceased probability:{probability[0][1]:.4f}")
