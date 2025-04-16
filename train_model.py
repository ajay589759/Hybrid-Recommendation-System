import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("Earthquake_of_last_30_days.csv")

# Selecting features and target
features = ["latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms", "net", "magType", "place", "type"]
target = "status"

# Handling missing values
df.fillna(0, inplace=True)

# Encoding categorical features
encoders = {}
for col in ["net", "magType", "place", "type"]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Encoding the target variable
target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

# Splitting dataset
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model and preprocessing objects
with open("earthquake_model.pkl", "wb") as f:
    pickle.dump((model, features, encoders, target_encoder, scaler), f)

print("Model trained and saved successfully.")
