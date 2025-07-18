import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Used to handle Cross-Origin Resource Sharing

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing your React app to connect

# --- Model and Scaler Loading ---
# Define paths to your saved model and scaler
# IMPORTANT: Adjust these paths if your 'models' directory is located elsewhere
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'xgb_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.joblib')

# Load the trained model and scaler globally when the app starts
try:
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Scaler loaded from: {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Model or scaler file not found. Ensure '{SCALER_PATH}' and '{MODEL_PATH}' exist.")
    loaded_scaler = None
    loaded_model = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    loaded_scaler = None
    loaded_model = None

# --- Feature Engineering Function (Identical to your notebook) ---
# This function MUST be exactly the same as the one used during training
def perform_feature_engineering(df):
    df_processed = df.copy()
    
    # Ensure required columns exist, fill with 0 if not (or handle as appropriate)
    for col in ['is_accident_burst', 'is_high_amount_accident', 'is_mistyped_entity_accident']:
        if col not in df_processed.columns:
            df_processed[col] = 0 # Assume no accident if column is missing
        df_processed[col] = df_processed[col].astype(int)

    # Convert timestamp
    if 'timestamp' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    else:
        # If timestamp is missing, create a dummy one or raise error
        df_processed['timestamp'] = pd.to_datetime('2024-01-01') # Placeholder

    # Sort by user and timestamp for time-based features
    # Ensure 'customer_id' is present
    if 'customer_id' not in df_processed.columns:
        df_processed['customer_id'] = 'default_customer' # Placeholder if missing
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])
    
    # Set timestamp as index for rolling operations (temporary)
    df_processed.set_index('timestamp', inplace=True)
    grouped = df_processed.groupby('customer_id')

    # Rolling features
    df_processed['mean_24hr'] = grouped['amount'].rolling('24H').mean().reset_index(level=0, drop=True)
    df_processed['stddev_30day'] = grouped['amount'].rolling('30D').std().reset_index(level=0, drop=True)
    df_processed['max_amount_7day'] = grouped['amount'].rolling('7D').max().reset_index(level=0, drop=True)
    df_processed.reset_index(inplace=True)

    # Extract time-based features
    df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['timestamp'].dt.day
    df_processed['month_of_year'] = df_processed['timestamp'].dt.month
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5,6]).astype(int)

    # Calculate time since last transaction
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])
    df_processed['time_since_last_txn'] = df_processed.groupby('customer_id')['timestamp'].diff().dt.total_seconds()
    df_processed['time_since_last_txn'].fillna(-1, inplace=True)

    # Rolling amount per recipient
    # Ensure 'recipient_entity' is present
    if 'recipient_entity' not in df_processed.columns:
        df_processed['recipient_entity'] = 'default_recipient' # Placeholder if missing
    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    df_processed.set_index('timestamp', inplace=True)
    grouped_recipient = df_processed.groupby(['customer_id', 'recipient_entity'])
    rolling_mean = grouped_recipient['amount'].rolling(window=30, min_periods=1).mean().shift(1)
    rolling_std = grouped_recipient['amount'].rolling(window=30, min_periods=1).std().shift(1)
    
    # Handle potential MultiIndex after rolling
    if isinstance(rolling_mean.index, pd.MultiIndex):
        rolling_mean.index = rolling_mean.index.droplevel([0,1])
    if isinstance(rolling_std.index, pd.MultiIndex):
        rolling_std.index = rolling_std.index.droplevel([0,1])

    df_processed['mean_amount_rolling'] = rolling_mean
    df_processed['std_amount_rolling'] = rolling_std
    
    # Robust handling for division by zero for amount_deviation_rolling
    temp_std = df_processed['std_amount_rolling'].replace(0, np.nan)
    df_processed['amount_deviation_rolling'] = (
        (df_processed['amount'] - df_processed['mean_amount_rolling']) / temp_std
    )
    df_processed['amount_deviation_rolling'].fillna(0, inplace=True)
    df_processed['amount_deviation_rolling'].replace([np.inf, -np.inf], 0, inplace=True)
    df_processed.reset_index(inplace=True)

    # Is new recipient
    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    first_transfer = df_processed.groupby(['customer_id', 'recipient_entity'])['timestamp'].min().reset_index()
    df_processed = df_processed.merge(first_transfer, on=['customer_id', 'recipient_entity'], suffixes=('', '_first'))
    df_processed['is_new_recipient'] = (df_processed['timestamp'] == df_processed['timestamp_first']).astype(int)
    df_processed.drop(columns=['timestamp_first'], inplace=True)

    # Duplicate transaction within X minutes
    X_minutes = 1
    df_processed = df_processed.sort_values(['customer_id', 'timestamp'])
    df_processed['recipient_amount_temp'] = df_processed['recipient_entity'].astype(str) + '_' + df_processed['amount'].astype(str)
    grouped_dup = df_processed.groupby(['customer_id', 'recipient_amount_temp'])
    df_processed['time_diff'] = grouped_dup['timestamp'].diff().dt.total_seconds().div(60)
    df_processed['is_duplicate_transaction_within_X_minutes'] = ((df_processed['time_diff'] <= X_minutes) & (df_processed['time_diff'] > 0)).astype(int)
    df_processed.drop(columns=['recipient_amount_temp', 'time_diff'], inplace=True)

    # Time deviation from user norm
    df_processed['seconds_since_midnight'] = df_processed['timestamp'].dt.hour * 3600 + df_processed['timestamp'].dt.minute * 60 + df_processed['timestamp'].dt.second
    mean_time = df_processed.groupby(['customer_id', 'recipient_entity'])['seconds_since_midnight'].transform(lambda x: x.shift().expanding().mean())
    df_processed['time_deviation_from_user_norm'] = (df_processed['seconds_since_midnight'] - mean_time).abs()
    df_processed.drop(columns=['seconds_since_midnight'], inplace=True)

    # Is unusual location for user
    # Ensure 'city' is present
    if 'city' not in df_processed.columns:
        df_processed['city'] = 'default_city' # Placeholder if missing

    df_processed = df_processed.sort_values(['customer_id', 'timestamp'])
    df_processed['is_unusual_location_for_user'] = 0
    seen_locations = {}
    for idx, row in df_processed.iterrows():
        user = row['customer_id']
        location = row['city']
        if user not in seen_locations:
            seen_locations[user] = set()
        if location not in seen_locations[user]:
            df_processed.at[idx, 'is_unusual_location_for_user'] = 1
            seen_locations[user].add(location)

    # This target column is not used for prediction but is kept for consistency
    df_processed['target_accident'] = (
        df_processed['is_accident_burst'].fillna(0).astype(int) |
        df_processed['is_high_amount_accident'].fillna(0).astype(int) |
        df_processed['is_mistyped_entity_accident'].fillna(0).astype(int)
    ).astype(int)

    return df_processed

# Define the exact feature columns used during training
feature_columns = [
    'amount',
    'mean_24hr', 'stddev_30day', 'max_amount_7day', 'hour_of_day', 'day_of_week',
    'day_of_month', 'month_of_year', 'is_weekend', 'time_since_last_txn',
    'mean_amount_rolling', 'std_amount_rolling', 'amount_deviation_rolling',
    'is_new_recipient', 'is_duplicate_transaction_within_X_minutes',
    'time_deviation_from_user_norm', 'is_unusual_location_for_user'
]

# --- Flask Route for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV directly into a DataFrame
            # Using io.StringIO to read from the file stream directly
            df = pd.read_csv(file)

            # Perform feature engineering
            df_engineered = perform_feature_engineering(df.copy())

            # Prepare features for prediction
            # Ensure only the feature columns are selected and handle NaNs/Infs
            X_predict = df_engineered[feature_columns].copy()
            
            # Drop rows where any of the critical features are NaN (from rolling calculations, etc.)
            # Keep track of original indices to map predictions back
            original_indices_for_prediction = X_predict.index
            X_predict.dropna(inplace=True)
            
            # Fill any remaining NaNs or infinities with 0 (or a suitable value)
            X_predict.fillna(0, inplace=True)
            X_predict.replace([np.inf, -np.inf], 0, inplace=True)

            # Filter the engineered DataFrame to match the rows that will be predicted
            df_filtered_for_prediction = df_engineered.loc[X_predict.index].copy()

            # Scale the features
            X_new_scaled = loaded_scaler.transform(X_predict)

            # Make predictions
            predictions = loaded_model.predict(X_new_scaled)
            prediction_proba = loaded_model.predict_proba(X_new_scaled)[:, 1]

            # Add predictions to the filtered DataFrame
            df_filtered_for_prediction['predicted_accident'] = predictions
            df_filtered_for_prediction['accident_probability'] = prediction_proba

            # Filter for accidental transactions
            accidental_transactions = df_filtered_for_prediction[df_filtered_for_prediction['predicted_accident'] == 1]

            # Prepare data for JSON response
            # Select relevant columns for the frontend display
            output_cols = ['timestamp', 'customer_id', 'amount', 'recipient_entity', 'city', 'accident_probability']
            
            # Convert timestamp to string for JSON serialization
            accidental_transactions_list = accidental_transactions[output_cols].to_dict(orient='records')
            
            # Ensure timestamp is string for JSON compatibility
            for item in accidental_transactions_list:
                if isinstance(item.get('timestamp'), pd.Timestamp):
                    item['timestamp'] = item['timestamp'].isoformat()

            return jsonify({"accidental_transactions": accidental_transactions_list})

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({"error": f"Error processing file: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

# --- How to Run the Flask App ---
if __name__ == '__main__':
    # This will run the Flask app on http://127.0.0.1:5000/
    # In a production environment, you would use a WSGI server like Gunicorn or uWSGI
    app.run(debug=True, host='0.0.0.0', port=5000)
