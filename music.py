import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
dataset_path = "C:/Users/ajays/OneDrive/Desktop/projects/music_recommendation_dataset.csv"
model_path = "C:/Users/ajays/OneDrive/Desktop/projects/music_model.pkl"


# Load dataset
def load_data():
    df = pd.read_csv("C:/Users/ajays/OneDrive/Desktop/projects/music_recommendation_dataset.csv")
    df = df[['Song', 'Artist', 'Genres']].dropna()
    df['Combined'] = df['Song'] + " " + df['Artist'] + " " + df['Genres']
    return df


# Train the model
def train_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    song_matrix = vectorizer.fit_transform(df['Combined'])
    similarity_matrix = cosine_similarity(song_matrix, song_matrix)
    with open(model_path, "wb") as f:
        pickle.dump((vectorizer, similarity_matrix), f)
    return vectorizer, similarity_matrix


# Load or train model
def load_or_train_model(df):
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            st.warning("Model corrupted. Retraining...")
            return train_model(df)
    else:
        return train_model(df)


# Get recommendations
def recommend_songs(song_name, df, vectorizer, similarity_matrix):
    if song_name not in df['Song'].values:
        return ["Song not found in database"]

    idx = df[df['Song'] == song_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [df.iloc[i[0]]['Song'] for i in scores]
    return recommendations


# Streamlit UI
def main():
    st.title("🎵 Music Recommendation System")
    df = load_data()
    vectorizer, similarity_matrix = load_or_train_model(df)

    song_name = st.text_input("Enter a song name:")
    if st.button("Recommend"):
        recommendations = recommend_songs(song_name, df, vectorizer, similarity_matrix)
        st.write("### Recommended Songs:")
        for song in recommendations:
            st.write(f"- {song}")


if __name__ == "__main__":
    main()