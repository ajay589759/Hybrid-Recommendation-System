import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Database setup
conn = sqlite3.connect('pricing_app.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS pricing_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Number_of_Riders INTEGER,
    Number_of_Drivers INTEGER,
    Location_Category TEXT,
    Customer_Loyalty_Status TEXT,
    Number_of_Past_Rides INTEGER,
    Average_Ratings FLOAT,
    Time_of_Booking TEXT,
    Vehicle_Type TEXT,
    Expected_Ride_Duration INTEGER,
    Prediction FLOAT
)''')
conn.commit()

# Load dataset and train model (only if needed)
@st.cache_resource
def train_and_save_model():
    df = pd.read_csv("dynamic_pricing.csv")

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop('Historical_Cost_of_Ride', axis=1)
    y = df['Historical_Cost_of_Ride']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning (fast RandomizedSearchCV)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Train best model
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = max(0, r2 * 100)  # Ensure non-negative accuracy

    # Save the model and preprocessors
    joblib.dump(best_model, 'pricing_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return r2, mse, accuracy

# Train model if not already trained
r2_score_value, mse_value, accuracy_rate = train_and_save_model()

# Load trained model and encoders
model = joblib.load('pricing_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit App
st.title("🚖 Dynamic Pricing Prediction App")
st.write(f"✅ **Model Accuracy Rate:** {accuracy_rate:.2f}%")
st.write(f"📊 **R² Score:** {r2_score_value:.2f}")
st.write(f"📉 **Mean Squared Error:** {mse_value:.2f}")

# Input form
st.header("Enter Ride Details")
number_of_riders = st.number_input("Number of Riders", min_value=0, step=1)
number_of_drivers = st.number_input("Number of Drivers", min_value=0, step=1)
location_category = st.selectbox("Location Category", ['Urban', 'Suburban', 'Rural'])
customer_loyalty_status = st.selectbox("Customer Loyalty Status", ['Regular', 'Silver', 'Gold'])
number_of_past_rides = st.number_input("Number of Past Rides", min_value=0, step=1)
average_ratings = st.number_input("Average Ratings", min_value=0.0, max_value=5.0, step=0.1)
time_of_booking = st.selectbox("Time of Booking", ['Morning', 'Afternoon', 'Evening', 'Night'])
vehicle_type = st.selectbox("Vehicle Type", ['Economy', 'Premium', 'Luxury'])
expected_ride_duration = st.number_input("Expected Ride Duration (mins)", min_value=1, step=1)

if st.button("Predict"):
    try:
        # Encode categorical inputs
        location_encoded = label_encoders['Location_Category'].transform([location_category])[0]
        loyalty_encoded = label_encoders['Customer_Loyalty_Status'].transform([customer_loyalty_status])[0]
        time_encoded = label_encoders['Time_of_Booking'].transform([time_of_booking])[0]
        vehicle_encoded = label_encoders['Vehicle_Type'].transform([vehicle_type])[0]

        input_data = np.array([[
            number_of_riders, number_of_drivers, location_encoded, loyalty_encoded,
            number_of_past_rides, average_ratings, time_encoded, vehicle_encoded,
            expected_ride_duration
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Store user input and prediction
        cursor.execute('''INSERT INTO pricing_records (
            Number_of_Riders, Number_of_Drivers, Location_Category,
            Customer_Loyalty_Status, Number_of_Past_Rides, Average_Ratings,
            Time_of_Booking, Vehicle_Type, Expected_Ride_Duration, Prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (number_of_riders, number_of_drivers, location_category,
                        customer_loyalty_status, number_of_past_rides, average_ratings,
                        time_of_booking, vehicle_type, expected_ride_duration, prediction[0]))
        conn.commit()

        st.success(f"🚕 The predicted ride cost is: **${prediction[0]:.2f}**")
    except ValueError as e:
        st.error(f"Error: {str(e)}")

# View all records
if st.button("View All Records"):
    cursor.execute("SELECT * FROM pricing_records")
    records = cursor.fetchall()
    st.write(records)
