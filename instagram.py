import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import sqlite3
import joblib
import os

# Initialize database connection
conn = sqlite3.connect('instagram_prediction.db', check_same_thread=False)
cursor = conn.cursor()

# Check if 'saves' column exists, if not, add it
cursor.execute("PRAGMA table_info(insta_records)")
columns = [col[1] for col in cursor.fetchall()]
if 'saves' not in columns:
    cursor.execute("ALTER TABLE insta_records ADD COLUMN saves INTEGER;")
    conn.commit()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS insta_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    likes INTEGER,
    comments INTEGER,
    shares INTEGER,
    saves INTEGER,
    prediction FLOAT
)''')
conn.commit()

# Function to preprocess and train the model
@st.cache_resource
def train_and_save_model():
    file_path = "Instagram_data.csv"
    if not os.path.exists(file_path):
        st.error("Dataset file not found. Please upload 'Instagram_data.csv'.")
        return None

    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Check if necessary columns exist
    required_columns = ['Likes', 'Comments', 'Shares', 'Saves', 'Impressions']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
        return None

    # Calculate Engagement Rate
    df['Engagement_Rate'] = (df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']) / df['Impressions']

    X = df[['Likes', 'Comments', 'Shares', 'Saves']]
    y = df['Engagement_Rate']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)

    # Save model and scaler
    joblib.dump(model, 'insta_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return error

# Train and save the model if files are missing
if not os.path.exists('insta_model.pkl') or not os.path.exists('scaler.pkl'):
    error = train_and_save_model()
else:
    error = None

@st.cache_resource
def load_model():
    if os.path.exists('insta_model.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('insta_model.pkl'), joblib.load('scaler.pkl')
    else:
        st.error("Model files not found. Train the model first.")
        return None, None

model, scaler = load_model()

# Streamlit App
st.title("Instagram Engagement Prediction App")
if error is not None:
    st.write(f"Model MAE (Lower is better): {error:.4f}")

st.header("Enter Instagram Post Data")
likes = st.number_input("Likes", min_value=0, step=1)
comments = st.number_input("Comments", min_value=0, step=1)
shares = st.number_input("Shares", min_value=0, step=1)
saves = st.number_input("Saves", min_value=0, step=1)

if st.button("Predict Engagement Rate"):
    if model is None or scaler is None:
        st.error("Model not loaded. Train the model first.")
    else:
        input_data = np.array([[likes, comments, shares, saves]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Save input and prediction
        cursor.execute('''INSERT INTO insta_records (likes, comments, shares, saves, prediction) VALUES (?, ?, ?, ?, ?)''',
                       (likes, comments, shares, saves, float(prediction[0])))
        conn.commit()

        st.success(f"Predicted Engagement Rate: {prediction[0]:.4f}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM insta_records")
    records = cursor.fetchall()
    st.write(records)

# Close DB connection when script stops
conn.close()
