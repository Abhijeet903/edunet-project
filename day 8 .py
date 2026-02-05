import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.title("Welcome to energy prediction app")
model = jb.load("energy_appliance_model.pkl")
temp = st.number_input("Enter the temperature",min_value=0.0, max_value=45.0, value=5.0, step=0.1)

if st.button("Predict Energy Consumption"):
    new_data = np.array([[temp]])
    prediction = model.predict(new_data)
    st.write(f"Predicted Energy Consumption: {prediction[0]:.2f} kWh")
    st.write("Thank you for using the energy prediction app!")