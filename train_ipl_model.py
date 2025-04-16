import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
matches = pd.read_csv("matches.csv")

# Select features
features = ["team1", "team2", "toss_winner", "venue"]
X = matches[features].copy()  # Create a copy to avoid SettingWithCopyWarning
y = matches["winner"]

# Encode categorical features
label_encoders = {col: LabelEncoder().fit(X[col]) for col in features}
for col in features:
    X.loc[:, col] = label_encoders[col].transform(X[col])  # Fix warning

# Encode target
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model, encoders, and feature list
with open("ipl_model.pkl", "wb") as f:
    pickle.dump((model, label_encoders, y_encoder, features), f)

print("Model trained and saved successfully!")
