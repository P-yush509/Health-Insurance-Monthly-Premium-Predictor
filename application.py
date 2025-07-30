import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the entire pipeline (model + preprocessor)
try:
    with open('model.pkl', 'rb') as file:
        pipeline = joblib.load(file)
except (FileNotFoundError, AttributeError, ImportError) as e:
    st.error("⚠️ Error loading the model.")
    st.stop()

st.title("🏥 Health Insurance Monthly Premium Predictor")

# User input
age = st.slider("Age", 18, 100, 30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "southwest", "northwest"])

if st.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        'sex': sex,
        'smoker': smoker,
        'region': region,
        'age': age,
        'bmi': bmi,
        'children': children
    }])

    # Predict
    prediction = pipeline.predict(input_df)[0]

    if prediction <= 0:
        st.error("You don't need insurance based on the given inputs.")
    else:
        st.success(f"Estimated Monthly Premium: INR {prediction:.2f}")
