import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model  = joblib.load(os.path.join(BASE_DIR, "models", "breast_cancer_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

FEATURE_NAMES = [
    "age_at_diagnosis", "tumor_size", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "mutation_count", "nottingham_prognostic_index",
    "er_status", "her2_status", "pr_status", "chemotherapy",
    "hormone_therapy", "radio_therapy", "type_of_breast_surgery", "inferred_menopausal_state"
]

st.set_page_config(page_title="Breast Cancer Survival Predictor", layout="centered")

st.title("Breast Cancer Survival Prediction")
st.write("Enter patient clinical data to predict **Overall Survival Status** (Living or Deceased).")
st.caption("Based on the METABRIC dataset — 2,000+ breast cancer patients.")

st.sidebar.header("Patient Clinical Data")

def user_input():
    age_at_diagnosis              = st.sidebar.number_input("Age at Diagnosis", 20.0, 100.0, 60.0, step=0.5)
    tumor_size                    = st.sidebar.number_input("Tumor Size (mm)", 0.0, 200.0, 25.0, step=0.5)
    neoplasm_histologic_grade     = st.sidebar.selectbox("Neoplasm Histologic Grade", [1, 2, 3], index=1)
    lymph_nodes_examined_positive = st.sidebar.number_input("Lymph Nodes Examined Positive", 0, 50, 0)
    mutation_count                = st.sidebar.number_input("Mutation Count", 0, 200, 2)
    nottingham_prognostic_index   = st.sidebar.number_input("Nottingham Prognostic Index", 0.0, 10.0, 4.0, step=0.01)

    st.sidebar.markdown("---")
    er_status        = st.sidebar.selectbox("ER Status",                 ["Positive", "Negative"])
    her2_status      = st.sidebar.selectbox("HER2 Status",               ["Negative", "Positive"])
    pr_status        = st.sidebar.selectbox("PR Status",                 ["Positive", "Negative"])
    chemotherapy     = st.sidebar.selectbox("Chemotherapy",              ["No", "Yes"])
    hormone_therapy  = st.sidebar.selectbox("Hormone Therapy",           ["Yes", "No"])
    radio_therapy    = st.sidebar.selectbox("Radio Therapy",             ["Yes", "No"])
    surgery_type     = st.sidebar.selectbox("Type of Breast Surgery",    ["Breast Conserving", "Mastectomy"])
    menopausal_state = st.sidebar.selectbox("Inferred Menopausal State", ["Post", "Pre"])

    return pd.DataFrame([[
        age_at_diagnosis,
        tumor_size,
        float(neoplasm_histologic_grade),
        float(lymph_nodes_examined_positive),
        float(mutation_count),
        nottingham_prognostic_index,
        1.0 if er_status == "Positive" else 0.0,
        1.0 if her2_status == "Positive" else 0.0,
        1.0 if pr_status == "Positive" else 0.0,
        1.0 if chemotherapy == "Yes" else 0.0,
        1.0 if hormone_therapy == "Yes" else 0.0,
        1.0 if radio_therapy == "Yes" else 0.0,
        1.0 if surgery_type == "Mastectomy" else 0.0,
        1.0 if menopausal_state == "Post" else 0.0,
    ]], columns=FEATURE_NAMES)

input_data = user_input()

if st.button("Predict Survival"):
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)
    probability  = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Predicted: Deceased")
    else:
        st.success("Predicted: Living")

    st.write(f"Confidence: **{np.max(probability) * 100:.2f}%**")
    st.write({
        "Living Probability":   round(float(probability[0][0]), 4),
        "Deceased Probability": round(float(probability[0][1]), 4),
    })

st.subheader("Input Summary")
st.dataframe(input_data)

st.warning("This is a machine learning prediction and not a medical diagnosis.")
