import pandas as pd

def preprocess_input(match_data, label_encoders, expected_features):
    """
    Preprocesses user input for IPL match prediction.

    Parameters:
        match_data (DataFrame): The input match details dataframe.
        label_encoders (dict): Dictionary of trained label encoders.
        expected_features (list): List of features used during model training.

    Returns:
        DataFrame: Encoded match data ready for model prediction.
    """
    # Ensure only relevant features are passed
    match_data = match_data[expected_features]  # Drop extra features

    # Encode categorical columns using trained label encoders
    for col in match_data.columns:
        if col in label_encoders:
            match_data[col] = match_data[col].map(lambda x: label_encoders[col].transform([x])[0])

    return match_data
