import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize database connection
conn = sqlite3.connect('movie_recommendation.db')
cursor = conn.cursor()

# Create table for storing user inputs and recommendations
cursor.execute('''CREATE TABLE IF NOT EXISTS movie_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_movie TEXT,
    recommended_movies TEXT
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_and_save_model():
    file_path = "movie_recommendation_dataset.csv"
    if not os.path.exists(file_path):
        st.error("Dataset file not found. Please upload 'movie_recommendation_dataset.csv'.")
        return None

    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Check if necessary columns exist
    required_columns = ['original_title', 'genre']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {set(required_columns) - set(df.columns)}")
        return None

    # Fill NaN values
    df['genre'] = df['genre'].fillna('')

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save model and similarity matrix
    joblib.dump(df, 'movie_data.pkl')
    joblib.dump(similarity_matrix, 'similarity_matrix.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    return "Model trained successfully."


# Train and save the model if files are missing
if not os.path.exists('movie_data.pkl') or not os.path.exists('similarity_matrix.pkl'):
    train_and_save_model()


@st.cache_resource
def load_model():
    if os.path.exists('movie_data.pkl') and os.path.exists('similarity_matrix.pkl'):
        return joblib.load('movie_data.pkl'), joblib.load('similarity_matrix.pkl')
    else:
        st.error("Model files not found. Train the model first.")
        return None, None


df, similarity_matrix = load_model()

# Streamlit App
st.title("Movie Recommendation System")

st.header("Enter a Movie Title")
user_movie = st.text_input("Movie Title")

if st.button("Get Recommendations"):
    if df is None or similarity_matrix is None:
        st.error("Model not loaded. Train the model first.")
    else:
        if user_movie not in df['original_title'].values:
            st.error("Movie not found in database. Please try another title.")
        else:
            movie_idx = df[df['original_title'] == user_movie].index[0]
            similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_movies = [df.iloc[i[0]]['original_title'] for i in similarity_scores[1:6]]

            # Save input and recommendations
            cursor.execute('''INSERT INTO movie_records (user_movie, recommended_movies) VALUES (?, ?)''',
                           (user_movie, ", ".join(top_movies)))
            conn.commit()

            st.success(f"Top 5 Recommended Movies: {', '.join(top_movies)}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM movie_records")
    records = cursor.fetchall()
    st.write(records)