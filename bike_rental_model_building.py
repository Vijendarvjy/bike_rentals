import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

# 1. Load Data
data = pd.read_csv('hour.csv')

# 2. Feature Engineering
data['dteday'] = pd.to_datetime(data['dteday'])
data['day_of_week'] = data['dteday'].dt.dayofweek

def categorize_hour(hour):
    if 7 <= hour <= 9: return 'Morning Rush'
    elif 17 <= hour <= 19: return 'Evening Rush'
    elif 0 <= hour <= 5: return 'Low Demand'
    else: return 'Normal Hours'

data['time_category'] = data['hr'].apply(categorize_hour)

# 3. Preprocessing
data_processed = pd.get_dummies(data.copy(), columns=['season', 'weathersit', 'mnth', 'hr', 'weekday', 'time_category'], drop_first=True)
data_processed = data_processed.drop(columns=['instant', 'dteday'])

# 4. Split
split_index = int(len(data_processed) * 0.8)
train_df, test_df = data_processed.iloc[:split_index], data_processed.iloc[split_index:]

X_train = train_df.drop(columns=['cnt', 'casual', 'registered'])
y_train = train_df['cnt']

# 5. Hyperparameter Tuning (XGBoost)
param_dist = {
    'n_estimators': [200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.6, 0.8]
}

random_search = RandomizedSearchCV(XGBRegressor(random_state=42), param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_

# 6. Save Model
with open('tuned_xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

print('Model building script finished and model saved.')