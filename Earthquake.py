import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained model
def load_model():
    with open("earthquake_model.pkl", "rb") as f:
        model, feature_names, encoders, target_encoder, scaler = pickle.load(f)
    return model, feature_names, encoders, target_encoder, scaler

# Load the model
model, feature_names, encoders, target_encoder, scaler = load_model()

# Streamlit UI
st.title("Earthquake Prediction Model")

# User inputs
input_data = {}

for feature in feature_names:
    if feature in encoders:  # If categorical, use dropdown
        input_data[feature] = st.selectbox(f"Select {feature}:", encoders[feature].classes_)
    else:  # If numerical, use number input
        input_data[feature] = st.number_input(f"Enter {feature}:", step=0.01)

# Convert categorical inputs using the fitted encoders
for feature in encoders:
    input_data[feature] = encoders[feature].transform([input_data[feature]])[0]

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical features
input_df[feature_names] = scaler.transform(input_df[feature_names])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([prediction])[0]
    st.write(f"Predicted Earthquake Status: **{prediction_label}**")
