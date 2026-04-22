import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

VALID_PATIENT = {
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
}

HIGH_RISK_PATIENT = {
    "age_at_diagnosis": 78.0,
    "tumor_size": 65.0,
    "neoplasm_histologic_grade": 3.0,
    "lymph_nodes_examined_positive": 10.0,
    "mutation_count": 15.0,
    "nottingham_prognostic_index": 8.0,
    "er_status": 0,
    "her2_status": 1,
    "pr_status": 0,
    "chemotherapy": 1,
    "hormone_therapy": 0,
    "radio_therapy": 0,
    "type_of_breast_surgery": 1,
    "inferred_menopausal_state": 1,
}


def test_home():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert "predict" in body["endpoints"]


def test_predict_valid():
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("Living", "Deceased")
    assert body["risk_level"] in ("High", "Moderate", "Lower")
    assert "request_id" in body


def test_probabilities_sum_to_one():
    r = client.post("/predict", json=VALID_PATIENT)
    body = r.json()
    total = body["living_probability"] + body["deceased_probability"]
    assert abs(total - 1.0) < 1e-4


def test_confidence_score():
    r = client.post("/predict", json=VALID_PATIENT)
    body = r.json()
    expected = round(max(body["living_probability"], body["deceased_probability"]) * 100, 1)
    assert abs(body["confidence"] - expected) < 0.1


def test_high_risk_patient():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["deceased_probability"] > 0.5


def test_invalid_schema_422():
    r = client.post("/predict", json={"age_at_diagnosis": "not_a_number"})
    assert r.status_code == 422


def test_missing_fields_422():
    r = client.post("/predict", json={"age_at_diagnosis": 55.0})
    assert r.status_code == 422


def test_logs_endpoint():
    client.post("/predict", json=VALID_PATIENT)
    r = client.get("/logs?n=5")
    assert r.status_code == 200
    body = r.json()
    assert "records" in body
    assert isinstance(body["records"], list)
