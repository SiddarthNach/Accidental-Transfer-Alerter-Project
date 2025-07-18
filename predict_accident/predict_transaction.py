import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data (assuming the path is correct for your environment)
transaction_df = pd.read_csv("/Users/siddarthnachannagari/Accidental-Transfer-Alerter-Project/bank_transaction_data.csv")

# --- Feature Engineering Function ---
# Encapsulate all your feature engineering steps into a single function.
# This is crucial for consistency when applying to new data.
def perform_feature_engineering(df):
    # Make a copy to avoid SettingWithCopyWarning if input df is a slice
    df_processed = df.copy()

    # Convert accident flags to int
    df_processed['is_accident_burst'] = df_processed['is_accident_burst'].astype(int)
    df_processed['is_high_amount_accident'] = df_processed['is_high_amount_accident'].astype(int)
    df_processed['is_mistyped_entity_accident'] = df_processed['is_mistyped_entity_accident'].astype(int)

    # Convert timestamp
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])

    # Sort by user and timestamp for time-based features
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])

    # Set timestamp as index for rolling operations (temporary)
    df_processed.set_index('timestamp', inplace=True)

    # Group by user
    grouped = df_processed.groupby('customer_id')

    # 1. Mean transaction amount over past 24 hours
    df_processed['mean_24hr'] = grouped['amount'].rolling('24H').mean().reset_index(level=0, drop=True)

    # 2. Std deviation over past 30 days
    df_processed['stddev_30day'] = grouped['amount'].rolling('30D').std().reset_index(level=0, drop=True)

    # 3. Max amount over rolling 7-day window
    df_processed['max_amount_7day'] = grouped['amount'].rolling('7D').max().reset_index(level=0, drop=True)

    # Reset index back
    df_processed.reset_index(inplace=True)

    # Extract time-based features
    df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek   # Monday=0, Sunday=6
    df_processed['day_of_month'] = df_processed['timestamp'].dt.day
    df_processed['month_of_year'] = df_processed['timestamp'].dt.month

    # Create binary flag for weekend (Saturday=5, Sunday=6)
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5,6]).astype(int)

    # Sort data by user and timestamp to calculate time since last transaction
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])

    # Calculate time since last transaction for each user (in seconds)
    df_processed['time_since_last_txn'] = df_processed.groupby('customer_id')['timestamp'].diff().dt.total_seconds()
    df_processed['time_since_last_txn'].fillna(-1, inplace=True) # Fill NaN for first transaction

    # Rolling amount per recipient
    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    df_processed.set_index('timestamp', inplace=True) # Set index for rolling
    grouped_recipient = df_processed.groupby(['customer_id', 'recipient_entity'])
    rolling_mean = grouped_recipient['amount'].rolling(window=30, min_periods=1).mean().shift(1)
    rolling_std = grouped_recipient['amount'].rolling(window=30, min_periods=1).std().shift(1)

    # Align indices after rolling to correctly assign back
    rolling_mean.index = rolling_mean.index.droplevel([0,1])
    rolling_std.index = rolling_std.index.droplevel([0,1])

    df_processed['mean_amount_rolling'] = rolling_mean
    df_processed['std_amount_rolling'] = rolling_std

    df_processed['amount_deviation_rolling'] = (
        (df_processed['amount'] - df_processed['mean_amount_rolling']) / df_processed['std_amount_rolling']
    ).fillna(0) # Fillna for cases where std_amount_rolling is 0 or NaN
    df_processed.reset_index(inplace=True) # Reset index after rolling

    # Is new recipient
    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    first_transfer = df_processed.groupby(['customer_id', 'recipient_entity'])['timestamp'].min().reset_index()
    df_processed = df_processed.merge(first_transfer, on=['customer_id', 'recipient_entity'], suffixes=('', '_first'))
    df_processed['is_new_recipient'] = (df_processed['timestamp'] == df_processed['timestamp_first']).astype(int)
    df_processed.drop(columns=['timestamp_first'], inplace=True)

    # Duplicate transaction within X minutes
    X_minutes = 1 # Define the time window in minutes for duplicate detection
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
    df_processed.drop(columns=['seconds_since_midnight'], inplace=True) # Drop helper column

    # Is unusual location for user
    df_processed = df_processed.sort_values(['customer_id', 'timestamp'])
    df_processed['is_unusual_location_for_user'] = 0
    seen_locations = {}
    # Iterating row by row can be slow for large datasets.
    # For a production system, consider more optimized vectorized approaches if performance is critical.
    for idx, row in df_processed.iterrows():
        user = row['customer_id']
        location = row['city']
        if user not in seen_locations:
            seen_locations[user] = set()
        if location not in seen_locations[user]:
            # New location for this user
            df_processed.at[idx, 'is_unusual_location_for_user'] = 1
            seen_locations[user].add(location)

    # Create the combined target variable
    df_processed['target_accident'] = (
        df_processed['is_accident_burst'].fillna(0).astype(int) |
        df_processed['is_high_amount_accident'].fillna(0).astype(int) |
        df_processed['is_mistyped_entity_accident'].fillna(0).astype(int)
    ).astype(int)

    return df_processed

# Apply feature engineering to the raw data
transaction_df_engineered = perform_feature_engineering(transaction_df.copy()) # Pass a copy

# Define the feature columns that will go into the model
feature_columns = [
    'amount', # Don't forget 'amount' itself is a crucial feature
    'mean_24hr', 'stddev_30day', 'max_amount_7day', 'hour_of_day', 'day_of_week',
    'day_of_month', 'month_of_year', 'is_weekend', 'time_since_last_txn',
    'mean_amount_rolling', 'std_amount_rolling', 'amount_deviation_rolling',
    'is_new_recipient', 'is_duplicate_transaction_within_X_minutes',
    'time_deviation_from_user_norm', 'is_unusual_location_for_user'
]

# Prepare data for model training
# Drop rows where critical features might be NaN after engineering (e.g., first few rolling windows)
model_df = transaction_df_engineered.dropna(subset=feature_columns)

# Handle any remaining NaNs or infinities in the selected feature columns
model_df[feature_columns] = model_df[feature_columns].fillna(0) # Fill NaNs with 0 (or another appropriate value)
model_df[feature_columns].replace([np.inf, -np.inf], 0, inplace=True) # Replace infinities with 0

X = model_df[feature_columns]
y = model_df['target_accident']

X = X.fillna(0)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the XGBoost model
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Save the trained StandardScaler and the XGBoost model
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, "xgb_model.joblib")

print("StandardScaler saved successfully as 'scaler.joblib'")
print("XGBoost model saved successfully as 'xgb_model.joblib'")

# Optional: SHAP values (as in your original notebook)
# import shap
# explainer = shap.Explainer(model, X_train_scaled, feature_names=feature_columns)
# shap_values = explainer(X_test_scaled)
# shap.plots.beeswarm(shap_values)