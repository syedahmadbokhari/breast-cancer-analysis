import joblib
import numpy as np

# load saved model
model = joblib.load("models/breast_cancer_model.pkl")

# example patient sample (must match training features)
sample = np.array([[14.5, 20.3, 90.2, 600.5, 0.10, 0.15, 0.12, 0.05, 0.18, 0.07, 0.50, 0.20]])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Malignant tumor detected")
else:
    print("Benign tumor")