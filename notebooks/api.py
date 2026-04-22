from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Breast Cancer Survival Prediction API",
    description="Predicts Overall Survival Status using a tuned XGBoost pipeline trained on METABRIC.",
    version="2.0.0",
)

pipeline = joblib.load("models/pipeline.pkl")

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

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age_at_diagnosis": 65.0, "tumor_size": 28.0,
                "neoplasm_histologic_grade": 2.0, "lymph_nodes_examined_positive": 1.0,
                "mutation_count": 3.0, "nottingham_prognostic_index": 4.5,
                "er_status": 1, "her2_status": 0, "pr_status": 1,
                "chemotherapy": 0, "hormone_therapy": 1, "radio_therapy": 1,
                "type_of_breast_surgery": 0, "inferred_menopausal_state": 1
            }]
        }
    }


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Survival Prediction API",
        "model":   "XGBoost (Optuna-tuned) + StandardScaler Pipeline",
        "dataset": "METABRIC",
        "docs":    "/docs",
    }


@app.post("/predict")
def predict(data: PatientData):
    features = pd.DataFrame([[
        data.age_at_diagnosis, data.tumor_size, data.neoplasm_histologic_grade,
        data.lymph_nodes_examined_positive, data.mutation_count,
        data.nottingham_prognostic_index, data.er_status, data.her2_status,
        data.pr_status, data.chemotherapy, data.hormone_therapy,
        data.radio_therapy, data.type_of_breast_surgery, data.inferred_menopausal_state,
    ]], columns=FEATURE_NAMES)

    prediction  = pipeline.predict(features)
    probability = pipeline.predict_proba(features)

    return {
        "prediction":           "Deceased" if prediction[0] == 1 else "Living",
        "living_probability":   round(float(probability[0][0]), 4),
        "deceased_probability": round(float(probability[0][1]), 4),
        "confidence":           round(float(max(probability[0])) * 100, 2),
    }
