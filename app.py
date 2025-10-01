import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model (you can save your trained model using joblib or pickle)
# model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction")

# User inputs
area = st.number_input("Area (in sq ft):")
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, step=1)
location = st.selectbox("Location", ["Downtown", "Suburb", "City Center"])  # example

# Dummy prediction (replace with model.predict)
if st.button("Predict Price"):
    # Example input array
    input_data = np.array([[area, bedrooms]])
    # prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹{round(5000000 + area * 1000, 2)}")