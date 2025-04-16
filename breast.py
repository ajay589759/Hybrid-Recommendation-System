import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sqlite3
import joblib
import os

# Initialize database connection
conn = sqlite3.connect('breast_cancer_app.db')
cursor = conn.cursor()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS cancer_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    features TEXT,
    prediction TEXT
)''')
conn.commit()

@st.cache_resource
def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    df = pd.read_csv("breast-cancer.csv")

    # Check for missing values
    if df.isnull().sum().any():
        st.warning("Dataset contains missing values. Filling with median.")
        df.fillna(df.median(), inplace=True)

    # Map target variable to binary
    if 'diagnosis' not in df.columns:
        st.error("Target column 'diagnosis' is missing in the dataset.")
        st.stop()
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    return df

@st.cache_resource
def train_model():
    """Train the Random Forest model with GridSearchCV."""
    df = load_and_preprocess_data()

    # Features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Train the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

    # Save the model, scaler, and accuracy
    joblib.dump(best_model, 'cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(accuracy, 'accuracy.pkl')

    return accuracy, report

# Train the model if not already available
if not os.path.exists('cancer_model.pkl') or not os.path.exists('scaler.pkl') or not os.path.exists('accuracy.pkl'):
    accuracy, report = train_model()
else:
    model = joblib.load('cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    accuracy = joblib.load('accuracy.pkl')
    report = "Model already trained. Accuracy and metrics loaded."

# Load the model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.text(report)

# Input form
st.header("Enter Features for Prediction")

# Dynamically generate feature inputs
df = load_and_preprocess_data()
feature_cols = df.drop('diagnosis', axis=1).columns
inputs = {}
for feature in feature_cols:
    inputs[feature] = st.number_input(f"{feature}", step=0.01)

if st.button("Predict"):
    # Prepare input
    input_data = np.array([list(inputs.values())])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"

    # Save user input and prediction to the database
    cursor.execute('''INSERT INTO cancer_records (features, prediction) VALUES (?, ?)''',
                   (str(inputs), result))
    conn.commit()

    # Display the result
    st.success(f"The prediction is: {result}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM cancer_records")
    records = cursor.fetchall()
    st.write(records)