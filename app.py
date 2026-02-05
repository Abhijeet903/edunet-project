import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Urban Heat Risk AI", layout="wide")

st.title("🌍 AI-Based Urban Heat Island Risk Prediction System")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("green_heat_data.csv")

# -------------------- FEATURES --------------------
X = df.drop("heat_risk_score", axis=1)
y = df["heat_risk_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- MODEL TRAINING --------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lin_pred = lin_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# -------------------- PERFORMANCE --------------------
st.subheader("📈 Model Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("**Linear Regression**")
    st.write("R² Score:", round(r2_score(y_test, lin_pred), 3))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, lin_pred)), 2))

with col2:
    st.write("**Random Forest (Best Model)**")
    st.write("R² Score:", round(r2_score(y_test, rf_pred), 3))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2))

# -------------------- FEATURE IMPORTANCE --------------------
st.subheader("🔍 Feature Importance (Random Forest)")

importances = rf_model.feature_importances_
importance_df = pd.Series(importances, index=X.columns).sort_values()

fig, ax = plt.subplots()
importance_df.plot(kind='barh', ax=ax)
ax.set_title("Feature Contribution to Heat Risk")
st.pyplot(fig)

# -------------------- USER INPUT --------------------
st.header("🌡 Predict Heat Risk for an Area")

temp = st.slider("Temperature (°C)", 25, 45, 32)
lst = st.slider("Land Surface Temperature (°C)", 28, 48, 36)
aqi = st.slider("AQI", 70, 220, 130)
humidity = st.slider("Humidity (%)", 40, 75, 55)
pop_density = st.slider("Population Density (per km²)", 2500, 8000, 5000)
green_cover = st.slider("Green Cover (%)", 10, 50, 25)
built_up = st.slider("Built-up Area (%)", 35, 85, 65)

if st.button("Predict Heat Risk"):
    input_data = [[temp, lst, aqi, humidity, pop_density, green_cover, built_up]]
    risk = rf_model.predict(input_data)[0]

    st.success(f"🔥 Predicted Heat Risk Score: {round(risk, 2)} / 100")

    if risk < 45:
        st.info("Low risk zone. Maintain green infrastructure.")
    elif risk < 70:
        st.warning("Moderate risk. Increase plantation & reduce built-up heat surfaces.")
    else:
        st.error("High risk area! Immediate urban cooling strategies required.")

# -------------------- VISUAL ANALYSIS --------------------
st.subheader("📊 Temperature vs Heat Risk")

fig2, ax2 = plt.subplots()
ax2.scatter(df["temperature"], df["heat_risk_score"])
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Heat Risk Score")
st.pyplot(fig2)
