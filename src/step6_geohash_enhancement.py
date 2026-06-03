#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 6: GEOHASH ENHANCEMENT")
print("=" * 80)

X, y, X_test, feature_cols = get_processed_data()
train_raw = pd.read_csv('dataset/train.csv')

train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask].copy()
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask].copy()
y_val_raw = y[val_mask]

# Baseline Score (without new features)
model_base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model_base.fit(X_trn, y_trn_log)
preds_base = np.clip(np.expm1(model_base.predict(X_val)), 0, 1)
score_base = max(0, 100 * r2_score(y_val_raw, preds_base))
print(f"Baseline Score: {score_base:.4f}")

# --- Generate Leak-Safe Geohash Features ---
# We compute stats ONLY on Day 48 to prevent leakage into Day 49 Val
print("\nGenerating geohash-level statistics...")

# Add geohash and raw targets back temporarily for calculations
X_trn['geohash'] = train_raw.loc[train_mask, 'geohash'].values
X_trn['demand'] = y[train_mask]
X_val['geohash'] = train_raw.loc[val_mask, 'geohash'].values

# 1. Geohash-level Mean Demand
geo_demand = X_trn.groupby('geohash')['demand'].mean().to_dict()
X_trn['geo_mean_demand'] = X_trn['geohash'].map(geo_demand).fillna(X_trn['demand'].mean())
X_val['geo_mean_demand'] = X_val['geohash'].map(geo_demand).fillna(X_trn['demand'].mean())

# 2. Geohash-level Lane Statistics
if 'NumberofLanes_filled' in X_trn.columns:
    geo_lanes = X_trn.groupby('geohash')['NumberofLanes_filled'].agg(['mean', 'max'])
    geo_lanes.columns = ['geo_lanes_mean', 'geo_lanes_max']
    X_trn = X_trn.merge(geo_lanes, on='geohash', how='left')
    X_val = X_val.merge(geo_lanes, on='geohash', how='left')

# 3. Geohash-level Temperature Statistics
if 'Temperature_filled' in X_trn.columns:
    geo_temp = X_trn.groupby('geohash')['Temperature_filled'].agg(['mean', 'std'])
    geo_temp.columns = ['geo_temp_mean', 'geo_temp_std']
    # fillna for std if only 1 observation
    geo_temp = geo_temp.fillna(0)
    X_trn = X_trn.merge(geo_temp, on='geohash', how='left')
    X_val = X_val.merge(geo_temp, on='geohash', how='left')

# 4. Geohash-level Weather Distributions (fraction of time it is sunny, rainy, etc.)
if 'Weather_encoded' in X_trn.columns:
    weather_dummies = pd.get_dummies(X_trn['Weather_encoded'], prefix='geo_weather')
    weather_dummies['geohash'] = X_trn['geohash']
    geo_weather = weather_dummies.groupby('geohash').mean()
    
    X_trn = X_trn.merge(geo_weather, on='geohash', how='left')
    X_val = X_val.merge(geo_weather, on='geohash', how='left')

# Drop temporary columns
X_trn.drop(['geohash', 'demand'], axis=1, inplace=True)
X_val.drop(['geohash'], axis=1, inplace=True)

# Train model with new features
print("Training model with Geohash Enhancements...")
model_enh = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model_enh.fit(X_trn, y_trn_log)
preds_enh = np.clip(np.expm1(model_enh.predict(X_val)), 0, 1)
score_enh = max(0, 100 * r2_score(y_val_raw, preds_enh))

print(f"Enhanced Score: {score_enh:.4f}")
diff = score_enh - score_base
print(f"Difference:     {diff:+.4f} points")

if diff > 0.1:
    print("\nRECOMMENDATION: Keep geohash enhancement features. They significantly improve CV.")
elif diff > 0:
    print("\nRECOMMENDATION: Keep geohash enhancement features. Slight improvement.")
else:
    print("\nRECOMMENDATION: Discard geohash enhancement features or refine them. Score decreased.")

print("DONE.")
