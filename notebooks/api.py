import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "api.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PRED_LOG = LOG_DIR / "predictions.csv"
PRED_LOG_HEADER = [
    "timestamp", "request_id", "prediction", "living_probability",
    "deceased_probability", "confidence",
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count",
    "nottingham_prognostic_index", "er_status", "her2_status", "pr_status",
    "chemotherapy", "hormone_therapy", "radio_therapy",
    "type_of_breast_surgery", "inferred_menopausal_state",
]

if not PRED_LOG.exists():
    with open(PRED_LOG, "w", newline="") as f:
        csv.writer(f).writerow(PRED_LOG_HEADER)

# ── Model ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
pipeline = joblib.load(BASE_DIR / "models" / "pipeline.pkl")

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count",
    "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status",
    "chemotherapy", "hormone_therapy", "radio_therapy",
    "type_of_breast_surgery", "inferred_menopausal_state",
]

app = FastAPI(
    title="Breast Cancer Survival Predictor",
    description="XGBoost pipeline (Optuna-tuned) trained on METABRIC dataset",
    version="2.1.0",
)


class PatientData(BaseModel):
    age_at_diagnosis: float
    tumor_size: float
    neoplasm_histologic_grade: float
    lymph_nodes_examined_positive: float
    mutation_count: float
    nottingham_prognostic_index: float
    er_status: int
    her2_status: int
    pr_status: int
    chemotherapy: int
    hormone_therapy: int
    radio_therapy: int
    type_of_breast_surgery: int
    inferred_menopausal_state: int

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
                "inferred_menopausal_state": 1,
            }]
        }
    }


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Survival Predictor API",
        "model": "XGBoost (Optuna-tuned) — METABRIC dataset",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": {"predict": "POST /predict", "logs": "GET /logs"},
    }


@app.post("/predict")
def predict(data: PatientData):
    request_id = str(uuid.uuid4())[:8]
    try:
        features = pd.DataFrame(
            [[
                data.age_at_diagnosis, data.tumor_size,
                data.neoplasm_histologic_grade, data.lymph_nodes_examined_positive,
                data.mutation_count, data.nottingham_prognostic_index,
                data.er_status, data.her2_status, data.pr_status,
                data.chemotherapy, data.hormone_therapy, data.radio_therapy,
                data.type_of_breast_surgery, data.inferred_menopausal_state,
            ]],
            columns=FEATURE_NAMES,
        )

        prediction = pipeline.predict(features)[0]
        probability = pipeline.predict_proba(features)[0]

        living_prob   = round(float(probability[0]), 4)
        deceased_prob = round(float(probability[1]), 4)
        confidence    = round(max(living_prob, deceased_prob) * 100, 1)
        label         = "Deceased" if prediction == 1 else "Living"

        if deceased_prob >= 0.70:
            risk_level = "High"
        elif deceased_prob >= 0.50:
            risk_level = "Moderate"
        else:
            risk_level = "Lower"

        with open(PRED_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(), request_id, label,
                living_prob, deceased_prob, confidence,
                data.age_at_diagnosis, data.tumor_size,
                data.neoplasm_histologic_grade, data.lymph_nodes_examined_positive,
                data.mutation_count, data.nottingham_prognostic_index,
                data.er_status, data.her2_status, data.pr_status,
                data.chemotherapy, data.hormone_therapy, data.radio_therapy,
                data.type_of_breast_surgery, data.inferred_menopausal_state,
            ])

        logger.info(
            "request_id=%s prediction=%s confidence=%.1f%%",
            request_id, label, confidence,
        )

        return {
            "request_id": request_id,
            "prediction": label,
            "risk_level": risk_level,
            "living_probability": living_prob,
            "deceased_probability": deceased_prob,
            "confidence": confidence,
        }

    except Exception as exc:
        logger.error("request_id=%s error=%s", request_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/logs")
def get_logs(n: int = 20):
    """Return the last n prediction records."""
    if not PRED_LOG.exists():
        return {"records": []}
    with open(PRED_LOG, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"total": len(rows), "records": rows[-n:]}
