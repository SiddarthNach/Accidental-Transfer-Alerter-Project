#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


transaction_df = pd.read_csv("/Users/siddarthnachannagari/Accidental-Transfer-Alerter-Project/bank_transaction_data.csv")


# In[3]:


transaction_df.head(10)


# In[4]:


transaction_df['is_accident_burst'] = transaction_df['is_accident_burst'].astype(int)


# In[5]:


transaction_df['is_high_amount_accident'] = transaction_df['is_high_amount_accident'].astype(int)


# In[6]:


transaction_df['is_mistyped_entity_accident'] = transaction_df['is_mistyped_entity_accident'].astype(int)


# In[7]:


transaction_df.head(20)


# In[8]:


transaction_df.info()


# In[9]:


metrics = ['is_accident_burst', 'is_high_amount_accident', 'is_mistyped_entity_accident']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of Accident Metrics (0 vs 1)', fontsize=16)

for i, metric in enumerate(metrics):
    counts = transaction_df[metric].value_counts().sort_index()
    axes[i].bar(counts.index.astype(str), counts.values, color=['skyblue', 'salmon'])
    axes[i].set_title(metric)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['0 (No)', '1 (Yes)'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[10]:


transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])

# transactions per day
per_day = transaction_df.groupby(transaction_df['timestamp'].dt.date).size()

# transactions per week
per_week = transaction_df.groupby(transaction_df['timestamp'].dt.to_period('W')).size()

# transactions per user
per_user = transaction_df['customer_id'].value_counts()

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Transaction Frequency Analysis', fontsize=18)

# Plot per day
axes[0].plot(per_day.index, per_day.values, color='blue')
axes[0].set_title('Transactions per Day')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Count')

# Plot per week
axes[1].plot(per_week.index.astype(str), per_week.values, color='green')
axes[1].set_title('Transactions per Week')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Count')

# Plot per user (top 20 for readability)
axes[2].bar(per_user.head(20).index, per_user.head(20).values, color='orange')
axes[2].set_title('Top 20 Users by Number of Transactions')
axes[2].set_xlabel('Customer ID')
axes[2].set_ylabel('Transaction Count')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[11]:


import pandas as pd

# Ensure timestamp is datetime
transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])

# Sort by user and timestamp
transaction_df = transaction_df.sort_values(by=['customer_id', 'timestamp'])

# Set timestamp as index (needed for time-based rolling)
transaction_df.set_index('timestamp', inplace=True)

# Group by user
grouped = transaction_df.groupby('customer_id')

# 1. Mean transaction amount over past 24 hours
transaction_df['mean_24hr'] = grouped['amount'].rolling('24H').mean().reset_index(level=0, drop=True)

# 2. Std deviation over past 30 days
transaction_df['stddev_30day'] = grouped['amount'].rolling('30D').std().reset_index(level=0, drop=True)

# 3. Max amount over rolling 7-day window
transaction_df['max_amount_7day'] = grouped['amount'].rolling('7D').max().reset_index(level=0, drop=True)

# Reset index back if needed
transaction_df.reset_index(inplace=True)


# In[12]:


print(transaction_df[['customer_id', 'timestamp', 'amount', 'mean_24hr', 'stddev_30day', 'max_amount_7day']].head(10))


# In[13]:


# Drop rows with any NaNs in the feature columns
model_df = transaction_df.dropna(subset=['mean_24hr', 'stddev_30day', 'max_amount_7day'])


# In[14]:


print(model_df[['customer_id', 'timestamp', 'amount', 'mean_24hr', 'stddev_30day', 'max_amount_7day']].head())


# In[15]:


print(f"Original rows: {len(transaction_df)}")
print(f"After dropna: {len(model_df)}")


# In[16]:


print(transaction_df.columns.tolist())


# In[17]:


# Make sure timestamp is datetime
transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])

# Extract time-based features
transaction_df['hour_of_day'] = transaction_df['timestamp'].dt.hour
transaction_df['day_of_week'] = transaction_df['timestamp'].dt.dayofweek   # Monday=0, Sunday=6
transaction_df['day_of_month'] = transaction_df['timestamp'].dt.day
transaction_df['month_of_year'] = transaction_df['timestamp'].dt.month

# Create binary flag for weekend (Saturday=5, Sunday=6)
transaction_df['is_weekend'] = transaction_df['day_of_week'].isin([5,6]).astype(int)

# Sort data by user and timestamp to calculate time since last transaction
transaction_df = transaction_df.sort_values(by=['customer_id', 'timestamp'])

# Calculate time since last transaction for each user (in seconds)
transaction_df['time_since_last_txn'] = transaction_df.groupby('customer_id')['timestamp'].diff().dt.total_seconds()

# Optional: For first transaction per user, fill NaN with some value, e.g. -1 or 0
transaction_df['time_since_last_txn'].fillna(-1, inplace=True)

# Preview
print(transaction_df[['customer_id', 'timestamp', 'hour_of_day', 'day_of_week', 'is_weekend', 'time_since_last_txn']].head())


# In[18]:


transaction_df.head(10)


# In[19]:


transaction_df = transaction_df.sort_values(['customer_id', 'recipient_entity', 'timestamp'])
transaction_df.set_index('timestamp', inplace=True)

grouped = transaction_df.groupby(['customer_id', 'recipient_entity'])

rolling_mean = grouped['amount'].rolling(window=30, min_periods=1).mean().shift(1)
rolling_std = grouped['amount'].rolling(window=30, min_periods=1).std().shift(1)

# rolling_mean and rolling_std will have MultiIndex (customer_id, recipient_entity, timestamp)
# We want to assign these back keeping the original index

# Remove multiindex to match original DataFrame's index
rolling_mean.index = rolling_mean.index.droplevel([0,1])
rolling_std.index = rolling_std.index.droplevel([0,1])

transaction_df['mean_amount_rolling'] = rolling_mean
transaction_df['std_amount_rolling'] = rolling_std

# Calculate deviation
transaction_df['amount_deviation_rolling'] = (
    (transaction_df['amount'] - transaction_df['mean_amount_rolling']) / transaction_df['std_amount_rolling']
).fillna(0)

transaction_df.reset_index(inplace=True)


# In[20]:


print(transaction_df.columns.tolist())


# In[21]:


transaction_df.head(10)


# In[22]:


# Sort by user and timestamp
transaction_df = transaction_df.sort_values(['customer_id', 'timestamp'])

# Group by user and recipient and get first transaction timestamp per group
first_transfer = transaction_df.groupby(['customer_id', 'recipient_entity'])['timestamp'].min().reset_index()

# Merge back to original and flag if this transaction is the first time
transaction_df = transaction_df.merge(first_transfer, on=['customer_id', 'recipient_entity'], suffixes=('', '_first'))

transaction_df['is_new_recipient'] = (transaction_df['timestamp'] == transaction_df['timestamp_first']).astype(int)

# Drop helper column
transaction_df.drop(columns=['timestamp_first'], inplace=True)


# In[23]:


X = 1 # Define the time window in minutes for duplicate detection

# Sort by user and timestamp
transaction_df = transaction_df.sort_values(['customer_id', 'timestamp'])

# Create a helper column combining recipient and amount (to match duplicates)
transaction_df['recipient_amount'] = transaction_df['recipient_entity'].astype(str) + '_' + transaction_df['amount'].astype(str)

# Group by user and the combined column
grouped = transaction_df.groupby(['customer_id', 'recipient_amount'])

# Calculate time difference to previous transaction with same recipient+amount
transaction_df['time_diff'] = grouped['timestamp'].diff().dt.total_seconds().div(60)  # in minutes

# Flag duplicates within X minutes (exclude first transaction with NaN diff)
transaction_df['is_duplicate_transaction_within_X_minutes'] = ((transaction_df['time_diff'] <= X) & (transaction_df['time_diff'] > 0)).astype(int)

# Clean up helper columns if you want
transaction_df.drop(columns=['recipient_amount', 'time_diff'], inplace=True)


# In[24]:


# Convert timestamp to seconds since midnight
transaction_df['seconds_since_midnight'] = transaction_df['timestamp'].dt.hour * 3600 + transaction_df['timestamp'].dt.minute * 60 + transaction_df['timestamp'].dt.second

# Group by user and recipient, get mean scheduled time in seconds
mean_time = transaction_df.groupby(['customer_id', 'recipient_entity'])['seconds_since_midnight'].transform(lambda x: x.shift().expanding().mean())

# Calculate absolute deviation in seconds
transaction_df['time_deviation_from_user_norm'] = (transaction_df['seconds_since_midnight'] - mean_time).abs()

# Optionally drop helper column
transaction_df.drop(columns=['seconds_since_midnight'], inplace=True)


# In[25]:


# Sort by user and timestamp
transaction_df = transaction_df.sort_values(['customer_id', 'timestamp'])

# Group by user, collect unique locations seen before each transaction
transaction_df['is_unusual_location_for_user'] = 0

# We'll create a set of seen locations per user and mark if current location is new
seen_locations = {}

for idx, row in transaction_df.iterrows():
    user = row['customer_id']
    location = row['city']  
    
    if user not in seen_locations:
        seen_locations[user] = set()
    
    if location not in seen_locations[user]:
        # New location for this user
        transaction_df.at[idx, 'is_unusual_location_for_user'] = 1
        seen_locations[user].add(location)


# In[26]:


transaction_df.head(10)


# In[27]:


model_df = transaction_df.dropna()


# In[28]:


print(model_df.columns.tolist())


# In[29]:


model_df.head(10)


# In[30]:


model_df['target_accident'] = (
    model_df['is_accident_burst'].fillna(0).astype(int) |
    model_df['is_high_amount_accident'].fillna(0).astype(int) |
    model_df['is_mistyped_entity_accident'].fillna(0).astype(int)
).astype(int)


# In[31]:


drop_cols = [
    'timestamp', 'customer_id', 'transaction_id',
    'recipient_entity', 'original_recipient_entity',
    'is_accident_burst', 'is_high_amount_accident', 'is_mistyped_entity_accident',
    'target_accident'  # exclude this from features
]

X = model_df.drop(columns=drop_cols)  # Keep 'transaction_type', 'city', 'entity_type'

categorical_cols = ['transaction_type', 'city', 'entity_type']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# In[32]:


print(model_df.columns)


# In[33]:


model_df['is_accident'] = (
    model_df['is_accident_burst'].fillna(0).astype(int) |
    model_df['is_high_amount_accident'].fillna(0).astype(int) |
    model_df['is_mistyped_entity_accident'].fillna(0).astype(int)
).astype(int)


# In[34]:


for label in model_df['is_accident'].unique():
    subset = model_df[model_df['is_accident'] == label]
    plt.scatter(subset['amount'], subset['time_since_last_txn'], label=f"Accident={label}")

plt.xlabel('Amount')
plt.ylabel('Time Since Last Txn')
plt.title('Transaction Clusters by Accident Label')
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


feature_columns = [
    'mean_24hr', 'stddev_30day', 'max_amount_7day', 'hour_of_day', 'day_of_week',
    'day_of_month', 'month_of_year', 'is_weekend', 'time_since_last_txn',
    'mean_amount_rolling', 'std_amount_rolling', 'amount_deviation_rolling',
    'is_new_recipient', 'is_duplicate_transaction_within_X_minutes',
    'time_deviation_from_user_norm', 'is_unusual_location_for_user'
]


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = model_df[feature_columns]
y = model_df['target_accident']

X = X.fillna(0)
X.replace([np.inf, -np.inf], np.nan, inplace=True)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[42]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
print(classification_report(y_test, y_pred))


# In[38]:


import shap

explainer = shap.Explainer(model, X_train_scaled, feature_names=feature_columns)
shap_values = explainer(X_test_scaled)
shap.plots.beeswarm(shap_values)


# In[ ]:


import joblib

joblib.dump(model, "xgb_model.joblib")
joblib.dump(pipeline, "pipeline.joblib")

