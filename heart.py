import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import sqlite3
import joblib

# Initialize database connection
conn = sqlite3.connect('heart_app.db')
cursor = conn.cursor()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS heart_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps FLOAT,
    chol FLOAT,
    fbs INTEGER,
    restecg INTEGER,
    thalach FLOAT,
    exang INTEGER,
    oldpeak FLOAT,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    prediction TEXT
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_model():
    # Load the dataset
    df = pd.read_csv("heart.csv")

    # Check for missing values and impute if necessary
    df.fillna(df.median(), inplace=True)

    # Features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Train the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Cross-validation accuracy
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()

    # Save the model and scaler
    joblib.dump(best_model, 'heart_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return accuracy


# Train the model
accuracy = train_model()
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("Heart Disease Prediction App")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Input form
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=1, step=1)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
chol = st.number_input("Cholesterol Level", min_value=0.0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0)
slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1)
thal = st.number_input("Thal (0-3)", min_value=0, max_value=3, step=1)

if st.button("Predict"):
    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    result = "Positive" if prediction[0] == 1 else "Negative"

    # Save user input and prediction to the database
    cursor.execute('''INSERT INTO heart_records (
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result))
    conn.commit()

    # Display the result
    st.success(f"The prediction is: {result}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM heart_records")
    records = cursor.fetchall()
    st.write(records)
