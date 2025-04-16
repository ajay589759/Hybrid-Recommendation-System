# fit_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Final_Updated.csv")
df = df.dropna()

selected_features = [
    'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
    'Compound', 'TyreLife', 'FreshTyre',
    'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
    'TrackTemp', 'WindDirection', 'WindSpeed'
]

target = 'LapTime_in_seconds'
df = df[selected_features + [target]]

# Encode categorical features
label_encoders = {}
for col in ['Compound', 'FreshTyre', 'WindDirection']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f"{col}_encoder.pkl")
    label_encoders[col] = le

X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(f"✅ Accuracy (R²): {r2_score(y_test, model.predict(X_test)):.2f}")
joblib.dump(model, "fit_model.pkl")
