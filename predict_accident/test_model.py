import pandas as pd
import numpy as np
import joblib

loaded_scaler = joblib.load("/Users/siddarthnachannagari/Accidental-Transfer-Alerter-Project/model/scaler.joblib")
loaded_model = joblib.load("/Users/siddarthnachannagari/Accidental-Transfer-Alerter-Project/model/xgb_model.joblib")

def perform_feature_engineering(df):
    df_processed = df.copy()
    df_processed['is_accident_burst'] = df_processed['is_accident_burst'].astype(int)
    df_processed['is_high_amount_accident'] = df_processed['is_high_amount_accident'].astype(int)
    df_processed['is_mistyped_entity_accident'] = df_processed['is_mistyped_entity_accident'].astype(int)
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])
    df_processed.set_index('timestamp', inplace=True)
    grouped = df_processed.groupby('customer_id')
    df_processed['mean_24hr'] = grouped['amount'].rolling('24H').mean().reset_index(level=0, drop=True)
    df_processed['stddev_30day'] = grouped['amount'].rolling('30D').std().reset_index(level=0, drop=True)
    df_processed['max_amount_7day'] = grouped['amount'].rolling('7D').max().reset_index(level=0, drop=True)
    df_processed.reset_index(inplace=True)
    df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
    df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['timestamp'].dt.day
    df_processed['month_of_year'] = df_processed['timestamp'].dt.month
    df_processed['is_weekend'] = df_processed['day_of_week'].isin([5,6]).astype(int)
    df_processed = df_processed.sort_values(by=['customer_id', 'timestamp'])
    df_processed['time_since_last_txn'] = df_processed.groupby('customer_id')['timestamp'].diff().dt.total_seconds()
    df_processed['time_since_last_txn'].fillna(-1, inplace=True)

    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    df_processed.set_index('timestamp', inplace=True)
    grouped_recipient = df_processed.groupby(['customer_id', 'recipient_entity'])
    rolling_mean = grouped_recipient['amount'].rolling(window=30, min_periods=1).mean().shift(1)
    rolling_std = grouped_recipient['amount'].rolling(window=30, min_periods=1).std().shift(1)
    rolling_mean.index = rolling_mean.index.droplevel([0,1])
    rolling_std.index = rolling_std.index.droplevel([0,1])
    df_processed['mean_amount_rolling'] = rolling_mean
    df_processed['std_amount_rolling'] = rolling_std
    df_processed['amount_deviation_rolling'] = (
        (df_processed['amount'] - df_processed['mean_amount_rolling']) / df_processed['std_amount_rolling']
    ).fillna(0)
    df_processed.reset_index(inplace=True)

    df_processed = df_processed.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
    first_transfer = df_processed.groupby(['customer_id', 'recipient_entity'])['timestamp'].min().reset_index()
    df_processed = df_processed.merge(first_transfer, on=['customer_id', 'recipient_entity'], suffixes=('', '_first'))
    df_processed['is_new_recipient'] = (df_processed['timestamp'] == df_processed['timestamp_first']).astype(int)
    df_processed.drop(columns=['timestamp_first'], inplace=True)

    X_minutes = 1
    df_processed = df_processed.sort_values(['customer_id', 'timestamp'])
    df_processed['recipient_amount_temp'] = df_processed['recipient_entity'].astype(str) + '_' + df_processed['amount'].astype(str)
    grouped_dup = df_processed.groupby(['customer_id', 'recipient_amount_temp'])
    df_processed['time_diff'] = grouped_dup['timestamp'].diff().dt.total_seconds().div(60)
    df_processed['is_duplicate_transaction_within_X_minutes'] = ((df_processed['time_diff'] <= X_minutes) & (df_processed['time_diff'] > 0)).astype(int)
    df_processed.drop(columns=['recipient_amount_temp', 'time_diff'], inplace=True)

    df_processed['seconds_since_midnight'] = df_processed['timestamp'].dt.hour * 3600 + df_processed['timestamp'].dt.minute * 60 + df_processed['timestamp'].dt.second
    mean_time = df_processed.groupby(['customer_id', 'recipient_entity'])['seconds_since_midnight'].transform(lambda x: x.shift().expanding().mean())
    df_processed['time_deviation_from_user_norm'] = (df_processed['seconds_since_midnight'] - mean_time).abs()
    df_processed.drop(columns=['seconds_since_midnight'], inplace=True)

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

    df_processed['target_accident'] = (
        df_processed['is_accident_burst'].fillna(0).astype(int) |
        df_processed['is_high_amount_accident'].fillna(0).astype(int) |
        df_processed['is_mistyped_entity_accident'].fillna(0).astype(int)
    ).astype(int)

    return df_processed

feature_columns = [
    'amount',
    'mean_24hr', 'stddev_30day', 'max_amount_7day', 'hour_of_day', 'day_of_week',
    'day_of_month', 'month_of_year', 'is_weekend', 'time_since_last_txn',
    'mean_amount_rolling', 'std_amount_rolling', 'amount_deviation_rolling',
    'is_new_recipient', 'is_duplicate_transaction_within_X_minutes',
    'time_deviation_from_user_norm', 'is_unusual_location_for_user'
]

def predict_accidents_from_csv(csv_filepath):
    new_raw_df = pd.read_csv(csv_filepath)
    df_for_prediction = new_raw_df.copy()

    df_engineered = perform_feature_engineering(df_for_prediction)

    X_predict = df_engineered[feature_columns].copy()

    original_indices = X_predict.index
    X_predict.dropna(inplace=True)
    
    X_predict.fillna(0, inplace=True)
    X_predict.replace([np.inf, -np.inf], 0, inplace=True)

    df_engineered_filtered = df_engineered.loc[X_predict.index].copy()

    X_new_scaled = loaded_scaler.transform(X_predict)

    predictions = loaded_model.predict(X_new_scaled)
    prediction_proba = loaded_model.predict_proba(X_new_scaled)[:, 1]

    df_engineered_filtered['predicted_accident'] = predictions
    df_engineered_filtered['accident_probability'] = prediction_proba

    return df_engineered_filtered

new_csv_path = "/Users/siddarthnachannagari/Accidental-Transfer-Alerter-Project/predict_accident/mock_synthetic_data.csv"

try:
    results_df = predict_accidents_from_csv(new_csv_path)

    validation_cols = [
        'timestamp', 'customer_id', 'amount', 'recipient_entity', 'city',
        'is_accident_burst', 'is_high_amount_accident', 'is_mistyped_entity_accident',
        'target_accident',
        'predicted_accident', 'accident_probability'
    ]

    print("\nPrediction Results (first 10 rows with predictions and true labels):")
    print(results_df[validation_cols].head(10))

    print("\nTransactions predicted as accidental (with true labels):")
    accidental_transactions_df = results_df[results_df['predicted_accident'] == 1]
    if not accidental_transactions_df.empty:
        print(accidental_transactions_df[validation_cols].head())
        print(f"\nTotal accidental transactions predicted: {len(accidental_transactions_df)}")

        actual_accidents = results_df[results_df['target_accident'] == 1]
        correctly_predicted_accidents = actual_accidents[actual_accidents['predicted_accident'] == 1]
        print(f"Total actual accidents in this dataset: {len(actual_accidents)}")
        print(f"Actual accidents correctly predicted: {len(correctly_predicted_accidents)}")
        print(f"Recall (True Positives / All Actual Positives): {len(correctly_predicted_accidents) / len(actual_accidents) if len(actual_accidents) > 0 else 0:.4f}")

    else:
        print("No accidental transactions predicted in this file.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure the CSV file exists and contains the necessary columns for feature engineering.")