#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TIME-SERIES FORECASTING STRATEGY")
print("=" * 80)

# 1. Load Data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

def parse_time(ts):
    h, m = map(int, ts.split(':'))
    return h * 4 + m // 15

train['time_idx'] = train['timestamp'].apply(parse_time)
test['time_idx'] = test['timestamp'].apply(parse_time)

# 2. Build Day 48 Temporal Profiles (The Template)
d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

# Base template: Exact match
template_exact = d48.groupby(['geohash', 'time_idx'])['demand'].mean().reset_index()
template_exact.rename(columns={'demand': 'd48_lag_demand'}, inplace=True)

# RoadType / Grouped template (fallback)
d48['RoadType'] = d48['RoadType'].fillna('MISSING')
template_road = d48.groupby(['RoadType', 'time_idx'])['demand'].mean().reset_index()
template_road.rename(columns={'demand': 'd48_road_demand'}, inplace=True)

# Rolling Smoothing on Day 48
d48_sorted = d48.sort_values(['geohash', 'time_idx'])
d48_sorted['d48_rolling_mean'] = d48_sorted.groupby('geohash')['demand'].transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())
template_rolling = d48_sorted[['geohash', 'time_idx', 'd48_rolling_mean']]

# Combine Templates
templates = pd.merge(template_exact, template_rolling, on=['geohash', 'time_idx'], how='outer')

# 3. Extract Day 49 Real-Time Trend (Times 0 to 8)
# Compute average demand per geohash for Day 48 (time 0-8) and Day 49 (time 0-8)
early_48 = d48[d48['time_idx'] <= 8].groupby('geohash')['demand'].mean().reset_index()
early_49 = d49[d49['time_idx'] <= 8].groupby('geohash')['demand'].mean().reset_index()
early_48.rename(columns={'demand': 'early_d48_mean'}, inplace=True)
early_49.rename(columns={'demand': 'early_d49_mean'}, inplace=True)

trends = pd.merge(early_48, early_49, on='geohash', how='left')
# For missing Day 49 early trends, use global median trend
global_trend_ratio = (trends['early_d49_mean'] / trends['early_d48_mean'].replace(0, np.nan)).median()
trends['trend_multiplier'] = (trends['early_d49_mean'] / trends['early_d48_mean'].replace(0, np.nan)).fillna(global_trend_ratio)
trends['trend_offset'] = (trends['early_d49_mean'] - trends['early_d48_mean']).fillna(0)

# Clip extreme multipliers to prevent blow-ups
trends['trend_multiplier'] = trends['trend_multiplier'].clip(0.1, 10.0)

# 4. Feature Engineering function
def engineer_ts_features(df_input):
    df = df_input.copy()
    df['RoadType'] = df['RoadType'].fillna('MISSING')
    
    # Merge exact templates
    df = pd.merge(df, templates, on=['geohash', 'time_idx'], how='left')
    # Merge road templates
    df = pd.merge(df, template_road, on=['RoadType', 'time_idx'], how='left')
    # Merge trends
    df = pd.merge(df, trends[['geohash', 'trend_multiplier', 'trend_offset']], on='geohash', how='left')
    
    # Fill NAs
    df['trend_multiplier'] = df['trend_multiplier'].fillna(global_trend_ratio)
    df['trend_offset'] = df['trend_offset'].fillna(0)
    df['d48_lag_demand'] = df['d48_lag_demand'].fillna(df['d48_road_demand'])
    df['d48_rolling_mean'] = df['d48_rolling_mean'].fillna(df['d48_road_demand'])
    
    # Create Heuristic Expected Demand
    df['heuristic_demand'] = df['d48_rolling_mean'] * df['trend_multiplier']
    
    return df

# 5. Build Training Set using Day 49
# We will train the ML model on the early hours of Day 49, so it learns how to combine the templates and trends
train_ts = engineer_ts_features(d49)

features = ['d48_lag_demand', 'd48_rolling_mean', 'd48_road_demand', 
            'trend_multiplier', 'trend_offset', 'heuristic_demand',
            'NumberofLanes', 'time_idx']
target = 'demand'

# Local Validation: Train on time 0-5, Valid on time 6-8
tr_mask = train_ts['time_idx'] <= 5
vl_mask = train_ts['time_idx'] > 5

X_train, y_train = train_ts[tr_mask][features], train_ts[tr_mask][target]
X_valid, y_valid = train_ts[vl_mask][features], train_ts[vl_mask][target]

# Heuristic Baseline Score
val_heuristic = train_ts[vl_mask]['heuristic_demand']
score_heuristic = max(0, 100 * r2_score(y_valid, val_heuristic))
print(f"Time-Series Heuristic Baseline R² (Time 6-8): {score_heuristic:.4f}")

# 6. Train ML Forecaster
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': 42
}
model = lgb.LGBMRegressor(**lgb_params, n_estimators=500)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

val_ml_preds = model.predict(X_valid)
score_ml = max(0, 100 * r2_score(y_valid, val_ml_preds))
print(f"Time-Series ML Forecaster R² (Time 6-8): {score_ml:.4f}")

# Refit on ALL Day 49 for final prediction
print("\nRefitting ML Forecaster on ALL Day 49 early signals...")
model_final = lgb.LGBMRegressor(**lgb_params, n_estimators=model.best_iteration_ or 100)
model_final.fit(train_ts[features], train_ts[target])

# 7. Generate Submission
test_ts = engineer_ts_features(test)
test_preds = model_final.predict(test_ts[features])

# Output
sub = test[['Index']].copy()
sub['demand'] = test_preds
sub.to_csv('submission_ts_forecast.csv', index=False)
print("Saved submission_ts_forecast.csv")
