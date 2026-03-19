import streamlit as st
import numpy as np
import joblib

# Load model + scaler
model = joblib.load("models/breast_cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Title
st.title("🧬 Breast Cancer Prediction App")
st.write("Enter tumor measurements to predict whether it is **Benign or Malignant**.")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input():
    radius_mean = st.sidebar.number_input("Radius Mean", 0.0, 50.0, 14.0)
    texture_mean = st.sidebar.number_input("Texture Mean", 0.0, 50.0, 20.0)
    perimeter_mean = st.sidebar.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
    area_mean = st.sidebar.number_input("Area Mean", 0.0, 2000.0, 600.0)
    smoothness_mean = st.sidebar.number_input("Smoothness Mean", 0.0, 1.0, 0.1)
    compactness_mean = st.sidebar.number_input("Compactness Mean", 0.0, 1.0, 0.15)
    concavity_mean = st.sidebar.number_input("Concavity Mean", 0.0, 1.0, 0.12)
    concave_points_mean = st.sidebar.number_input("Concave Points Mean", 0.0, 1.0, 0.05)
    symmetry_mean = st.sidebar.number_input("Symmetry Mean", 0.0, 1.0, 0.18)
    fractal_dimension_mean = st.sidebar.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.07)
    radius_se = st.sidebar.number_input("Radius SE", 0.0, 5.0, 0.5)
    texture_se = st.sidebar.number_input("Texture SE", 0.0, 5.0, 0.2)

    data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,
        radius_se,
        texture_se
    ]])

    return data

# Get input
input_data = user_input()

# Predict button
if st.button("🔍 Predict"):

    # ✅ Apply scaling (CRITICAL)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    # ✅ Correct label mapping (based on your dataset)
    if prediction[0] == 1:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor Detected")

    # Confidence
    st.write(f"Confidence: {np.max(probability)*100:.2f}%")

    # Show probabilities
    st.write({
        "Benign Probability": float(probability[0][0]),
        "Malignant Probability": float(probability[0][1])
    })

# Show input summary
st.subheader("Input Summary")
st.dataframe(input_data)

# Disclaimer
st.warning("⚠️ This is a machine learning prediction and not a medical diagnosis.")