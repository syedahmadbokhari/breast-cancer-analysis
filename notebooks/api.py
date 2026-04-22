from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model  = joblib.load("models/breast_cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
]


class PatientData(BaseModel):
    age_at_diagnosis: float
    tumor_size: float
    neoplasm_histologic_grade: float
    lymph_nodes_examined_positive: float
    mutation_count: float
    nottingham_prognostic_index: float
    er_status: float            # 1=Positive, 0=Negative
    her2_status: float          # 1=Positive, 0=Negative
    pr_status: float            # 1=Positive, 0=Negative
    chemotherapy: float         # 1=Yes, 0=No
    hormone_therapy: float      # 1=Yes, 0=No
    radio_therapy: float        # 1=Yes, 0=No
    type_of_breast_surgery: float     # 1=Mastectomy, 0=Breast Conserving
    inferred_menopausal_state: float  # 1=Post, 0=Pre


@app.get("/")
def home():
    return {"message": "Breast Cancer Survival Prediction API (METABRIC)"}


@app.post("/predict")
def predict(data: PatientData):
    features = pd.DataFrame([[
        data.age_at_diagnosis,
        data.tumor_size,
        data.neoplasm_histologic_grade,
        data.lymph_nodes_examined_positive,
        data.mutation_count,
        data.nottingham_prognostic_index,
        data.er_status,
        data.her2_status,
        data.pr_status,
        data.chemotherapy,
        data.hormone_therapy,
        data.radio_therapy,
        data.type_of_breast_surgery,
        data.inferred_menopausal_state,
    ]], columns=FEATURE_NAMES)

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)
    probability     = model.predict_proba(features_scaled)

    return {
        "prediction":           "Deceased" if prediction[0] == 1 else "Living",
        "living_probability":   round(float(probability[0][0]), 4),
        "deceased_probability": round(float(probability[0][1]), 4),
    }
