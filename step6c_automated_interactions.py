#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 6C: AUTOMATED FEATURE INTERACTION SEARCH")
print("=" * 80)

X, y, X_test, feature_cols = get_processed_data()
train_raw = pd.read_csv('dataset/train.csv')

train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask].copy()
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask].copy()
y_val_raw = y[val_mask]

# Baseline Score (without new interactions)
model_base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model_base.fit(X_trn, y_trn_log)
preds_base = np.clip(np.expm1(model_base.predict(X_val)), 0, 1)
score_base = max(0, 100 * r2_score(y_val_raw, preds_base))
print(f"Baseline Score: {score_base:.4f}")

# Generate Candidate Interactions
print("\nGenerating candidate interactions...")

# We need the original categorical features for proper grouping if not already present
X_trn['geohash'] = train_raw.loc[train_mask, 'geohash'].values
X_val['geohash'] = train_raw.loc[val_mask, 'geohash'].values
X_trn['demand'] = y[train_mask]

# 1. RoadType x Hour (categorical interaction)
X_trn['rt_x_hour'] = X_trn['RoadType_encoded'].astype(str) + "_" + X_trn['hour'].astype(str)
X_val['rt_x_hour'] = X_val['RoadType_encoded'].astype(str) + "_" + X_val['hour'].astype(str)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(list(X_trn['rt_x_hour']) + list(X_val['rt_x_hour']))
X_trn['rt_x_hour_le'] = le.transform(X_trn['rt_x_hour'])
X_val['rt_x_hour_le'] = le.transform(X_val['rt_x_hour'])

# 2. RoadType x NumberofLanes
X_trn['rt_x_lanes'] = X_trn['RoadType_encoded'].fillna(-1) * X_trn['NumberofLanes_filled']
X_val['rt_x_lanes'] = X_val['RoadType_encoded'].fillna(-1) * X_val['NumberofLanes_filled']

# 3. Weather x Hour
X_trn['weather_x_hour_num'] = X_trn['Weather_encoded'].fillna(-1) * X_trn['hour']
X_val['weather_x_hour_num'] = X_val['Weather_encoded'].fillna(-1) * X_val['hour']

# 4. Weather x Temperature
X_trn['weather_x_temp'] = X_trn['Weather_encoded'].fillna(-1) * X_trn['Temperature_filled']
X_val['weather_x_temp'] = X_val['Weather_encoded'].fillna(-1) * X_val['Temperature_filled']

# 5. Geohash x Hour (Leak-safe Mean Demand)
geo_hour_demand = X_trn.groupby(['geohash', 'hour'])['demand'].mean().to_dict()
X_trn['geo_hour_mean'] = X_trn.set_index(['geohash', 'hour']).index.map(geo_hour_demand)
X_val['geo_hour_mean'] = X_val.set_index(['geohash', 'hour']).index.map(geo_hour_demand)
global_mean = X_trn['demand'].mean()
X_trn['geo_hour_mean'] = X_trn['geo_hour_mean'].fillna(global_mean)
X_val['geo_hour_mean'] = X_val['geo_hour_mean'].fillna(global_mean)

# 6. LargeVehicles x NumberofLanes
X_trn['lv_x_lanes_num'] = X_trn['LargeVehicles_encoded'].fillna(-1) * X_trn['NumberofLanes_filled']
X_val['lv_x_lanes_num'] = X_val['LargeVehicles_encoded'].fillna(-1) * X_val['NumberofLanes_filled']

# 7. RushHour x RoadType
X_trn['rush_x_rt_num'] = X_trn['is_rush_hour'] * X_trn['RoadType_encoded'].fillna(-1)
X_val['rush_x_rt_num'] = X_val['is_rush_hour'] * X_val['RoadType_encoded'].fillna(-1)

# 8. RushHour x Geohash (Leak-safe Mean Demand)
geo_rush_demand = X_trn.groupby(['geohash', 'is_rush_hour'])['demand'].mean().to_dict()
X_trn['geo_rush_mean'] = X_trn.set_index(['geohash', 'is_rush_hour']).index.map(geo_rush_demand)
X_val['geo_rush_mean'] = X_val.set_index(['geohash', 'is_rush_hour']).index.map(geo_rush_demand)
X_trn['geo_rush_mean'] = X_trn['geo_rush_mean'].fillna(global_mean)
X_val['geo_rush_mean'] = X_val['geo_rush_mean'].fillna(global_mean)

# 9. Temperature x Hour
X_trn['temp_x_hour'] = X_trn['Temperature_filled'] * X_trn['hour']
X_val['temp_x_hour'] = X_val['Temperature_filled'] * X_val['hour']

# Cleanup
drop_cols = ['geohash', 'demand', 'rt_x_hour']
X_trn.drop(drop_cols, axis=1, inplace=True)
X_val.drop(['geohash', 'rt_x_hour'], axis=1, inplace=True)

# Train model with new features
print("Training model with Interaction Enhancements...")
model_int = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model_int.fit(X_trn, y_trn_log)
preds_int = np.clip(np.expm1(model_int.predict(X_val)), 0, 1)
score_int = max(0, 100 * r2_score(y_val_raw, preds_int))

print(f"Interaction Enhanced Score: {score_int:.4f}")
diff = score_int - score_base
print(f"Difference:                 {diff:+.4f} points")

print("\nInteraction Feature Importances (Gain):")
interaction_cols = ['rt_x_hour_le', 'rt_x_lanes', 'weather_x_hour_num', 'weather_x_temp', 
                    'geo_hour_mean', 'lv_x_lanes_num', 'rush_x_rt_num', 'geo_rush_mean', 'temp_x_hour']
imp_df = pd.DataFrame({'Feature': X_trn.columns, 'Gain': model_int.booster_.feature_importance(importance_type='gain')})
imp_df = imp_df.sort_values('Gain', ascending=False)
print(imp_df[imp_df['Feature'].isin(interaction_cols)])

if diff > 0.1:
    print("\nRECOMMENDATION: Keep interactions. Substantial CV improvement.")
elif diff > 0:
    print("\nRECOMMENDATION: Keep interactions. Slight improvement.")
else:
    print("\nRECOMMENDATION: Discard interaction features. Score decreased.")

print("DONE.")
