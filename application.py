import streamlit as st
import numpy as np
import pandas as pd
import joblib

try:
    with open('model.pkl', 'rb') as file:
        model, preprocessor = joblib.load(file)
except (FileNotFoundError, AttributeError, ImportError) as e:
    st.error("Error loading the model. Please ensure the 'model.pkl' file is present and compatible.")
    st.stop()

st.title("Health Insurance Monthly Premium Predictor")

age = st.slider("Age", 18, 100)
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Children", 0, 10)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "southwest", "northwest"])

if st.button("Predict"):
    user_df = pd.DataFrame([[sex, smoker, region, age, bmi, children]], 
                           columns=['sex', 'smoker', 'region', 'age', 'bmi', 'children'])
    X_input = preprocessor.transform(user_df)
    prediction = model.predict(X_input)[0]
    if prediction <= 0:
        st.error("You Don't Need Insurance")
    else:
        st.success(f"Estimated Medical Cost: INR {prediction:.2f}")

