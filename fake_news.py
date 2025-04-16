# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import sqlite3
# import joblib
#
# # Initialize database connection
# conn = sqlite3.connect('fake_news_app.db')
# cursor = conn.cursor()
#
# # Create table for storing user inputs and predictions
# cursor.execute('''CREATE TABLE IF NOT EXISTS fake_news_records (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     word_count INTEGER,
#     num_sentences INTEGER,
#     unique_words INTEGER,
#     avg_word_length FLOAT,
#     prediction INTEGER
# )''')
# conn.commit()
#
#
# # Function to preprocess and train the model
# @st.cache_resource
# def train_model():
#     df = pd.read_csv("Fake_News_Detection.csv")
#     X = df.drop(columns=["ID", "Label"])
#     y = df["Label"]
#
#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
#     # Hyperparameter tuning
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [4, 6, None],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
#     rf = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
#     grid_search.fit(X_train, y_train)
#
#     # Train the best model
#     best_model = grid_search.best_estimator_
#     best_model.fit(X_train, y_train)
#
#     # Evaluate model accuracy
#     y_pred = best_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred) * 100
#
#     # Save model and scaler
#     joblib.dump(best_model, 'fake_news_model.pkl')
#     joblib.dump(scaler, 'scaler.pkl')
#
#     return accuracy
#
#
# # Train model
# accuracy = train_model()
# model = joblib.load('fake_news_model.pkl')
# scaler = joblib.load('scaler.pkl')
#
# # Streamlit App
# st.title("Fake News Detection App")
# st.write(f"Model Accuracy: {accuracy:.2f}%")
#
# # Input form
# st.header("Enter News Statistics")
# word_count = st.number_input("Word Count", min_value=1, step=1)
# num_sentences = st.number_input("Number of Sentences", min_value=1, step=1)
# unique_words = st.number_input("Unique Words", min_value=1, step=1)
# avg_word_length = st.number_input("Average Word Length", min_value=0.1)
#
# if st.button("Predict"):
#     input_data = np.array([[word_count, num_sentences, unique_words, avg_word_length]])
#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)
#
#     # Save input and prediction
#     cursor.execute('''INSERT INTO fake_news_records (
#         word_count, num_sentences, unique_words, avg_word_length, prediction
#     ) VALUES (?, ?, ?, ?, ?)''', (word_count, num_sentences, unique_words, avg_word_length, int(prediction[0])))
#     conn.commit()
#
#     st.success(f"The news is classified as: {'Fake' if prediction[0] == 1 else 'Real'}")
#
# # View all records
# if st.button("View All Records"):
#     cursor.execute("SELECT * FROM fake_news_records")
#     records = cursor.fetchall()
#     st.write(records)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sqlite3
import joblib

# Initialize database connection
conn = sqlite3.connect('fake_news_app_v2.db')
cursor = conn.cursor()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS fake_news_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_count INTEGER,
    num_sentences INTEGER,
    unique_words INTEGER,
    avg_word_length FLOAT,
    prediction INTEGER
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("Fake_News_Detection.csv")
    X = df.drop(columns=["ID", "Label"])
    y = df["Label"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
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

    # Evaluate model accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Save model and scaler
    joblib.dump(best_model, 'fake_news_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return accuracy


# Train model
accuracy = train_model()
model = joblib.load('fake_news_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("Fake News Detection App")
st.write(f"Model Accuracy: {accuracy:.2f}%")

# Instructions for running on a different local server URL
st.info("To run this app on a different port, use the following command:")
st.code("streamlit run your_script.py --server.port 8502", language="bash")

# Input form
st.header("Enter News Statistics")
word_count = st.number_input("Word Count", min_value=1, step=1)
num_sentences = st.number_input("Number of Sentences", min_value=1, step=1)
unique_words = st.number_input("Unique Words", min_value=1, step=1)
avg_word_length = st.number_input("Average Word Length", min_value=0.1)

if st.button("Predict"):
    input_data = np.array([[word_count, num_sentences, unique_words, avg_word_length]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Save input and prediction
    cursor.execute('''INSERT INTO fake_news_records (
        word_count, num_sentences, unique_words, avg_word_length, prediction
    ) VALUES (?, ?, ?, ?, ?)''', (word_count, num_sentences, unique_words, avg_word_length, int(prediction[0])))
    conn.commit()

    st.success(f"The news is classified as: {'Fake' if prediction[0] == 1 else 'Real'}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM fake_news_records")
    records = cursor.fetchall()
    st.write(records)
