import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction System")

age = st.number_input("Age")

sex = st.selectbox(
    "Sex",
    [0,1]
)

chol = st.number_input(
    "Cholesterol"
)

thalach = st.number_input(
    "Max Heart Rate"
)

if st.button("Predict"):

    sample = np.array([
        [
            age,
            sex,
            0,
            120,
            chol,
            0,
            0,
            thalach,
            0,
            0,
            0,
            0,
            0
        ]
    ])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Heart Disease Risk")
    else:
        st.success("No Heart Disease")