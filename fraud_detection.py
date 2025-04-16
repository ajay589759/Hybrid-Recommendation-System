import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

st.title("🔍 Fraud Detection System")

# 📂 Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Loaded Successfully!")
    st.write(df.head())

    # 🎯 Ensure 'Fraud' column exists
    if 'Fraud' not in df.columns:
        st.error("⚠️ 'Fraud' column not found! Check dataset.")
        st.stop()

    # 🔹 Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoder for later decoding

    # Define Features (X) and Target (y)
    X = df.drop(columns=['Fraud'])
    y = df['Fraud']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalize Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    # Save model and scaler
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump((model, scaler, label_encoders), f)
    st.write("💾 Model Saved Successfully!")

    # 🎯 Fraud Prediction Section
    st.subheader("🔍 Predict Fraudulent Transactions")

    # User input form
    user_input = {}
    for col in X.columns:
        if col in categorical_cols:
            options = df[col].unique().tolist()
            user_input[col] = st.selectbox(f"Select {col}:", options)
        else:
            user_input[col] = st.number_input(f"Enter {col}:")

    if st.button("Predict Fraud"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical values safely
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1)  # Assign -1 for unknown
            else:
                input_df[col] = -1  # Default unknown category

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = "Fraudulent" if prediction == 1 else "Not Fraudulent"
        st.success(f"🚀 Prediction: {result}")
