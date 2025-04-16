import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
import joblib

# Initialize database connection
conn = sqlite3.connect('insurance_app.db')
cursor = conn.cursor()

# Create table for storing user inputs and predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS insurance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex TEXT,
    bmi FLOAT,
    children INTEGER,
    smoker TEXT,
    region TEXT,
    prediction FLOAT
)''')
conn.commit()


# Function to preprocess and train the model
@st.cache_resource
def train_model():
    # Load the dataset
    df = pd.read_csv("medical_insurance.csv")

    # Handle categorical features using LabelEncoder
    label_encoder_sex = LabelEncoder()
    label_encoder_sex.fit(df['sex'])
    df['sex'] = label_encoder_sex.transform(df['sex'])

    label_encoder_smoker = LabelEncoder()
    label_encoder_smoker.fit(df['smoker'])
    df['smoker'] = label_encoder_smoker.transform(df['smoker'])

    label_encoder_region = LabelEncoder()
    label_encoder_region.fit(df['region'])
    df['region'] = label_encoder_region.transform(df['region'])

    # Features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Train the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate model accuracy (R^2 and MSE)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the model, scaler, and label encoders
    joblib.dump(best_model, 'insurance_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder_sex, 'label_encoder_sex.pkl')
    joblib.dump(label_encoder_smoker, 'label_encoder_smoker.pkl')
    joblib.dump(label_encoder_region, 'label_encoder_region.pkl')

    return r2, mse


# Train the model and show accuracy
r2_score_value, mse = train_model()
model = joblib.load('insurance_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_sex = joblib.load('label_encoder_sex.pkl')
label_encoder_smoker = joblib.load('label_encoder_smoker.pkl')
label_encoder_region = joblib.load('label_encoder_region.pkl')

# Streamlit App
st.title("Medical Insurance Cost Prediction App")
st.write(f"Model R^2 Score: {r2_score_value:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")

# Input form
st.header("Enter User Details")
age = st.number_input("Age", min_value=1, step=1)
sex = st.selectbox("Sex", ['male', 'female'])
bmi = st.number_input("BMI", min_value=0.0)
children = st.number_input("Number of Children", min_value=0, step=1)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

if st.button("Predict"):
    try:
        # Ensure the input labels are valid and handle unseen labels
        sex_encoded = label_encoder_sex.transform([sex])[0] if sex in label_encoder_sex.classes_ else -1
        smoker_encoded = label_encoder_smoker.transform([smoker])[0] if smoker in label_encoder_smoker.classes_ else -1
        region_encoded = label_encoder_region.transform([region])[0] if region in label_encoder_region.classes_ else -1

        if sex_encoded == -1 or smoker_encoded == -1 or region_encoded == -1:
            st.error("Invalid input value detected. Please choose from available options.")
        else:
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)

            # Save user input and prediction to the database
            cursor.execute('''INSERT INTO insurance_records (
                age, sex, bmi, children, smoker, region, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (age, sex, bmi, children, smoker, region, prediction[0]))
            conn.commit()

            # Display the result
            st.success(f"The predicted insurance cost is: ${prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Error: {str(e)}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM insurance_records")
    records = cursor.fetchall()
    st.write(records)
