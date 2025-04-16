import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Initialize database connection
conn = sqlite3.connect('web_series_recommendation.db')
cursor = conn.cursor()

# Create table for storing user queries and recommendations
cursor.execute('''CREATE TABLE IF NOT EXISTS series_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    recommended_series TEXT
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_and_save_model():
    file_path = "All_Streaming_Shows.csv"
    if not os.path.exists(file_path):
        st.error("Dataset file not found. Please upload 'All_Streaming_Shows.csv'.")
        return None

    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Correct column names
    df.rename(columns={'ï»¿Series Title': 'Title', 'Genre': 'Genres'}, inplace=True)

    # Check required columns
    required_columns = ['Title', 'Genres']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
        return None

    # Fill missing values
    df.dropna(subset=['Title', 'Genres'], inplace=True)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    genre_matrix = vectorizer.fit_transform(df['Genres'])

    # Train Nearest Neighbors model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(genre_matrix)

    # Save model and vectorizer
    joblib.dump(model, 'knn_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    df.to_csv('cleaned_streaming_data.csv', index=False)

    return "Model trained successfully!"


# Train and save the model if files are missing
if not os.path.exists('knn_model.pkl') or not os.path.exists('vectorizer.pkl'):
    train_and_save_model()


@st.cache_resource
def load_model():
    if os.path.exists('knn_model.pkl') and os.path.exists('vectorizer.pkl'):
        return joblib.load('knn_model.pkl'), joblib.load('vectorizer.pkl')
    else:
        st.error("Model files not found. Train the model first.")
        return None, None


model, vectorizer = load_model()


def recommend_series(user_input, top_n=5):
    try:
        df = pd.read_csv('cleaned_streaming_data.csv')
        genre_matrix = vectorizer.transform([user_input])
        distances, indices = model.kneighbors(genre_matrix, n_neighbors=top_n)
        recommendations = df.iloc[indices[0]]['Title'].tolist()
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []


# Streamlit App
st.title("Web Series Recommendation App")

st.header("Enter a Genre or Series Title")
user_input = st.text_input("Genre or Series Title")

if st.button("Get Recommendations"):
    if model is None or vectorizer is None:
        st.error("Model not loaded. Train the model first.")
    else:
        recommendations = recommend_series(user_input)
        if recommendations:
            st.success("Recommended Web Series:")
            for rec in recommendations:
                st.write(f"- {rec}")

            # Save query and recommendation
            cursor.execute('''INSERT INTO series_records (user_input, recommended_series) VALUES (?, ?)''',
                           (user_input, ', '.join(recommendations)))
            conn.commit()

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM series_records")
    records = cursor.fetchall()
    st.write(records)