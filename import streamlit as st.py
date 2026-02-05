import streamlit as st
import pandas as pd
import numpy as np 
import joblib as jb
 
st.title("Titanic Survival Prediction")
model = jb.load ("lr_titanic_model.pkl")

pc = st.number_input("Enter Passenger Class (1, 2, or 3):")
age = st.number_input("Enter Age of Passenger:")
sp = st.number_input("Enter Number of Siblings/Spouses Aboard:")
pch = st.number_input("Enter Number of Parents/Children Aboard:")
fare = st.number_input("Enter Fare Paid by Passenger:")
gender = st.number_input("Enter Gender of Passenger (0 for Male, 1 for Female):")
input_data = np.array([[pc, age, sp, pch, fare, gender]])
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("The passenger is predicted to have survived.")
    else:
        st.error("The passenger is predicted to have not survived.")
