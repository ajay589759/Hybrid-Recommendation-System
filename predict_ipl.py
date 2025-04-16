import pickle
import pandas as pd
import streamlit as st

# Load trained model and encoders
with open("ipl_model.pkl", "rb") as f:
    model, label_encoders, y_encoder, expected_features = pickle.load(f)  # ✅ Corrected

# User input
st.title("IPL Winner Prediction")

team1 = st.selectbox("Select Team 1", label_encoders["team1"].classes_)
team2 = st.selectbox("Select Team 2", label_encoders["team2"].classes_)
toss_winner = st.selectbox("Select Toss Winner", label_encoders["toss_winner"].classes_)
venue = st.selectbox("Select Venue", label_encoders["venue"].classes_)

# Prepare input
match_data = pd.DataFrame([[team1, team2, toss_winner, venue]], columns=expected_features)

# Encode input
for col in expected_features:
    match_data[col] = label_encoders[col].transform(match_data[col])

# Predict
prediction = model.predict(match_data)
predicted_winner = y_encoder.inverse_transform(prediction)[0]

# Display Result
st.write(f"🏏 Predicted Winner: **{predicted_winner}**")
