# fifa_predictor.py

import streamlit as st
import joblib

# Load models
model_multi = joblib.load("model_multiclass.pkl")
model_bin = joblib.load("model_binary.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("⚽ FIFA Match Win Predictor")

st.markdown("Enter match details to predict the outcome.")

# User Input
rank_diff = st.number_input("Home Team FIFA Rank - Away Team FIFA Rank", value=0)
points_diff = st.number_input("Home Team Points - Away Team Points", value=0)
neutral = st.selectbox("Neutral Location?", ["No", "Yes"])

neutral = 1 if neutral == "Yes" else 0

# Model selection
model_type = st.radio("Select Model Type:", ["Binary (Win/Not Win)", "Multiclass (Win/Draw/Lose)"])

if st.button("Predict"):
    features = [[rank_diff, points_diff, neutral]]

    if model_type == "Binary (Win/Not Win)":
        pred = model_bin.predict(features)[0]
        st.success("✅ Prediction: Home Team will **Win**" if pred == 1 else "❌ Prediction: Home Team will **NOT Win**")
    else:
        pred = model_multi.predict(features)[0]
        result = label_encoder.inverse_transform([pred])[0]
        st.success(f"📊 Prediction: Home Team will **{result.upper()}**")
