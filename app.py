import streamlit as st
import joblib
import numpy as np

model = joblib.load("tourism_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Tourism Rating Prediction")

year = st.number_input("Year")
month = st.number_input("Month")

if st.button("Predict"):
    input_data = np.array([[year, month, 0,0,0,0]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.write(prediction)
