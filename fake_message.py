import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import sqlite3
import joblib

# Initialize database connection
conn = sqlite3.connect('spam_detection.db')
cursor = conn.cursor()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS spam_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT,
    prediction INTEGER
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("Fake_message.csv")
    df.columns = ["Label", "Message"]  # Adjust column names if necessary
    df['Label'] = df['Label'].map({'spam': 1, 'ham': 0})  # Convert labels

    X = df["Message"]
    y = df["Label"]

    # Text feature extraction
    vectorizer = TfidfVectorizer()
    X_transformed = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with RandomizedSearchCV for faster execution
    param_grid = {
        'n_estimators': [100, 200],  # Reduced options for speed
        'max_depth': [None, 10],  # Limited depth values
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = RandomizedSearchCV(rf, param_grid, n_iter=5, cv=3, scoring='accuracy', random_state=42)
    grid_search.fit(X_train, y_train)

    # Train the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Save model and vectorizer
    joblib.dump(best_model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return accuracy


# Train model
accuracy = train_model()
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App
st.title("Spam Detection App")
st.write(f"Model Accuracy: {accuracy:.2f}%")

st.header("Enter a Message to Check for Spam")
message = st.text_area("Message Content")

if st.button("Predict"):
    input_transformed = vectorizer.transform([message])
    prediction = model.predict(input_transformed)

    # Save input and prediction
    cursor.execute('''INSERT INTO spam_records (message, prediction) VALUES (?, ?)''', (message, int(prediction[0])))
    conn.commit()

    st.success(f"The message is classified as: {'Spam' if prediction[0] == 1 else 'Not Spam'}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM spam_records")
    records = cursor.fetchall()
    st.write(records)
