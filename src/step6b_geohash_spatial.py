#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import lightgbm as lgb
import pygeohash as pgh
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 6B: GEOHASH SPATIAL INTELLIGENCE")
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

# --- Decode Geohash to Lat/Lon ---
print("\nDecoding geohashes and generating spatial clusters...")
X_trn['geohash'] = train_raw.loc[train_mask, 'geohash'].values
X_val['geohash'] = train_raw.loc[val_mask, 'geohash'].values

def decode_geo(gh):
    if pd.isna(gh):
        return 0.0, 0.0
    lat, lon = pgh.decode(gh)
    return lat, lon

# Apply decoding
coords_trn = X_trn['geohash'].apply(decode_geo).tolist()
X_trn['lat'] = [c[0] for c in coords_trn]
X_trn['lon'] = [c[1] for c in coords_trn]

coords_val = X_val['geohash'].apply(decode_geo).tolist()
X_val['lat'] = [c[0] for c in coords_val]
X_val['lon'] = [c[1] for c in coords_val]

# Generate KMeans clusters based on Lat/Lon
kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
# Fit on train only to prevent leakage
X_trn['spatial_cluster_30'] = kmeans.fit_predict(X_trn[['lat', 'lon']])
X_val['spatial_cluster_30'] = kmeans.predict(X_val[['lat', 'lon']])

# Drop temporary geohash string
X_trn.drop(['geohash'], axis=1, inplace=True)
X_val.drop(['geohash'], axis=1, inplace=True)

# Train model with new features
print("Training model with Spatial Enhancements...")
model_spat = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model_spat.fit(X_trn, y_trn_log)
preds_spat = np.clip(np.expm1(model_spat.predict(X_val)), 0, 1)
score_spat = max(0, 100 * r2_score(y_val_raw, preds_spat))

print(f"Spatial Enhanced Score: {score_spat:.4f}")
diff = score_spat - score_base
print(f"Difference:             {diff:+.4f} points")

# Feature importances for spatial features
print("\nSpatial Feature Importances (Gain):")
imp_df = pd.DataFrame({'Feature': X_trn.columns, 'Gain': model_spat.booster_.feature_importance(importance_type='gain')})
imp_df = imp_df.sort_values('Gain', ascending=False)
print(imp_df[imp_df['Feature'].isin(['lat', 'lon', 'spatial_cluster_30'])])

if diff > 0.1:
    print("\nRECOMMENDATION: Keep spatial features (lat, lon, clusters). They improve CV.")
elif diff > 0:
    print("\nRECOMMENDATION: Keep spatial features. Slight improvement.")
else:
    print("\nRECOMMENDATION: Discard spatial features or refine them. Score decreased.")

print("DONE.")
