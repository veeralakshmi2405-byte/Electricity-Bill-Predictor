# app.py
import streamlit as st
import joblib
import json
import numpy as np

# Title
st.title("Electricity Bill Predictor 💡")

# Load trained model
try:
    model = joblib.load("electricity_bill_model.pkl")
    with open("model_features.json", "r") as f:
        features = json.load(f)
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model files not found. Make sure 'electricity_bill_model.pkl' and 'model_features.json' are in the repo root.")
    st.stop()

# Section: User Input
st.subheader("Enter appliance usage and monthly details:")

input_data = []
for f in features:
    val = st.number_input(f, min_value=0.0, value=100.0)
    input_data.append(val)

# Predict button
if st.button("Predict Electricity Bill"):
    arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    st.metric("Estimated Electricity Bill (₹)", f"{prediction:.2f}")
