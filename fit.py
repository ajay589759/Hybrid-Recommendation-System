import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("fit_model.pkl")
compound_encoder = joblib.load("Compound_encoder.pkl")
fresh_encoder = joblib.load("FreshTyre_encoder.pkl")
wind_encoder = joblib.load("WindDirection_encoder.pkl")

st.title("🏁 F1 Fit Score Predictor")
st.write("Predicts lap time (in seconds) as a Fit Score based on race inputs.")

# User Inputs
speed_i1 = st.number_input("Speed at I1", 100, 400, 300)
speed_i2 = st.number_input("Speed at I2", 100, 400, 310)
speed_fl = st.number_input("Fastest Lap Speed", 100, 400, 320)
speed_st = st.number_input("Speed at ST", 100, 400, 315)

compound = st.selectbox("Tyre Compound", compound_encoder.classes_)
tyrelife = st.number_input("Tyre Life", 0, 50, 5)
freshtyre = st.selectbox("Fresh Tyre", fresh_encoder.classes_)

air_temp = st.number_input("Air Temp", -10, 50, 25)
humidity = st.number_input("Humidity", 0, 100, 50)
pressure = st.number_input("Pressure", 900, 1100, 1013)
rainfall = st.number_input("Rainfall", 0.0, 100.0, 0.0)
track_temp = st.number_input("Track Temp", 0, 70, 35)
wind_dir = st.selectbox("Wind Direction", wind_encoder.classes_)
wind_speed = st.number_input("Wind Speed", 0, 100, 10)

# Encode inputs
compound_encoded = compound_encoder.transform([compound])[0]
freshtyre_encoded = fresh_encoder.transform([freshtyre])[0]
wind_encoded = wind_encoder.transform([wind_dir])[0]

input_data = np.array([[
    speed_i1, speed_i2, speed_fl, speed_st,
    compound_encoded, tyrelife, freshtyre_encoded,
    air_temp, humidity, pressure, rainfall,
    track_temp, wind_encoded, wind_speed
]])

# Prediction
if st.button("Predict Lap Time (Fit Score)"):
    pred = model.predict(input_data)[0]
    st.success(f"🏎️ Predicted Lap Time: **{pred:.2f} seconds**")
