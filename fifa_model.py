# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("international_matches.csv")

# Select essential features
df = df[[
    'home_team', 'away_team',
    'home_team_fifa_rank', 'away_team_fifa_rank',
    'home_team_total_fifa_points', 'away_team_total_fifa_points',
    'neutral_location', 'home_team_result'
]].dropna()

# Feature engineering
df['rank_diff'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['points_diff'] = df['home_team_total_fifa_points'] - df['away_team_total_fifa_points']
df['neutral_location'] = df['neutral_location'].astype(int)

# Encode target for multi-class classification
label_encoder = LabelEncoder()
df['result_encoded'] = label_encoder.fit_transform(df['home_team_result'])

# Binary target
df['home_team_win'] = (df['home_team_result'] == 'Win').astype(int)

# Features
X = df[['rank_diff', 'points_diff', 'neutral_location']]

# MULTI-CLASS MODEL
y_multi = df['result_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)
model_multi = RandomForestClassifier(n_estimators=200, random_state=42)
model_multi.fit(X_train, y_train)

# BINARY MODEL
y_binary = df['home_team_win']
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)
model_bin = RandomForestClassifier(n_estimators=200, random_state=42)
model_bin.fit(X_train_bin, y_train_bin)

# Save models
joblib.dump(model_multi, "model_multiclass.pkl")
joblib.dump(model_bin, "model_binary.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Models trained and saved!")
