#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
from final_fe_utils import get_final_data

print("=" * 80)
print("ROUND 2: STEP 8 & 9 - TARGET ENCODING EXPERIMENT")
print("=" * 80)

# Load data without the previous target encoding to test fresh
X, y, X_test, feature_cols = get_final_data()
train_raw = pd.read_csv('dataset/train.csv')
test_raw = pd.read_csv('dataset/test.csv')

train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask].copy()
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask].copy()
y_val_raw = y[val_mask]

# Step 8: Geohash Coverage
geo_trn = set(train_raw.loc[train_mask, 'geohash'].unique())
geo_tst = set(test_raw['geohash'].unique())
geo_val = set(train_raw.loc[val_mask, 'geohash'].unique())

unseen_tst = geo_tst - geo_trn
unseen_val = geo_val - geo_trn

print(f"Day 48 Train Geohashes: {len(geo_trn)}")
print(f"Test Geohashes:         {len(geo_tst)}")
print(f"Unseen in Test:         {len(unseen_tst)} ({(len(unseen_tst)/len(geo_tst))*100:.2f}%)")
print(f"Unseen in Validation:   {len(unseen_val)} ({(len(unseen_val)/len(geo_val))*100:.2f}%)")

X_trn['geohash'] = train_raw.loc[train_mask, 'geohash'].values
X_val['geohash'] = train_raw.loc[val_mask, 'geohash'].values
X_trn['demand'] = y[train_mask]

global_mean = X_trn['demand'].mean()
global_median = X_trn['demand'].median()

# Step 9: Encoding Experiments
print("\nRunning Target Encoding Experiments...")

def evaluate_encoding(name, val_col):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    
    # Train set features
    features = [c for c in X_trn.columns if c not in ['geohash', 'demand', 'geo_mean_demand']]
    X_trn_eval = X_trn[features].copy()
    X_val_eval = X_val[features].copy()
    
    # swap the target encoded column
    X_trn_eval['encoded_demand'] = X_trn[val_col]
    X_val_eval['encoded_demand'] = X_val[val_col]
    
    model.fit(X_trn_eval, y_trn_log)
    preds = np.clip(np.expm1(model.predict(X_val_eval)), 0, 1)
    return max(0, 100 * r2_score(y_val_raw, preds))

# 1. Mean Target Encoding (Current Baseline)
geo_mean = X_trn.groupby('geohash')['demand'].mean().to_dict()
X_trn['te_mean'] = X_trn['geohash'].map(geo_mean).fillna(global_mean)
X_val['te_mean'] = X_val['geohash'].map(geo_mean).fillna(global_mean)
score_mean = evaluate_encoding("Mean TE", "te_mean")
print(f"Mean TE Score: {score_mean:.4f}")

# 2. Median Target Encoding
geo_median = X_trn.groupby('geohash')['demand'].median().to_dict()
X_trn['te_median'] = X_trn['geohash'].map(geo_median).fillna(global_median)
X_val['te_median'] = X_val['geohash'].map(geo_median).fillna(global_median)
score_median = evaluate_encoding("Median TE", "te_median")
print(f"Median TE Score: {score_median:.4f}")

# 3. Smoothed Target Encoding
# formula: (global_mean * alpha + group_mean * count) / (alpha + count)
alpha = 10
geo_stats = X_trn.groupby('geohash')['demand'].agg(['mean', 'count'])
geo_smoothed = ((global_mean * alpha) + (geo_stats['mean'] * geo_stats['count'])) / (alpha + geo_stats['count'])
geo_smoothed_dict = geo_smoothed.to_dict()

X_trn['te_smooth'] = X_trn['geohash'].map(geo_smoothed_dict).fillna(global_mean)
X_val['te_smooth'] = X_val['geohash'].map(geo_smoothed_dict).fillna(global_mean)
score_smooth = evaluate_encoding("Smoothed TE", "te_smooth")
print(f"Smoothed TE Score: {score_smooth:.4f}")

best_score = max(score_mean, score_median, score_smooth)
if best_score == score_smooth:
    best_enc = "Smoothed TE"
elif best_score == score_median:
    best_enc = "Median TE"
else:
    best_enc = "Mean TE"
    
print(f"\nBest Encoding: {best_enc} ({best_score:.4f})")

# Write a modified final_fe_utils file called r2_fe_utils.py that bakes in the best method and native categorical option
fe_utils_code = f"""import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pygeohash as pgh
from sklearn.cluster import KMeans

def parse_timestamp(df):
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = ts[0]
    df['minute'] = ts[1]
    return df

def engineer_features(df):
    df = df.copy()
    df = parse_timestamp(df)
    
    for i in range(min(6, df['geohash'].str.len().min())):
        df[f'geohash_char_{{i}}'] = df['geohash'].str[i]
    df['geohash_prefix_5'] = df['geohash'].str[:5]
    
    roadtype_map = {{'Residential': 0, 'Commercial': 1, 'Highway': 2}}
    df['RoadType_encoded'] = df['RoadType'].map(roadtype_map)
    df['RoadType_missing'] = df['RoadType'].isnull().astype(int)
    
    lv_map = {{'Not Allowed': 0, 'Allowed': 1}}
    df['LargeVehicles_encoded'] = df['LargeVehicles'].map(lv_map)
    
    landmarks_map = {{'No': 0, 'Yes': 1}}
    df['Landmarks_encoded'] = df['Landmarks'].map(landmarks_map)
    
    df['n_missing'] = df[['RoadType', 'Temperature', 'Weather', 'Landmarks', 'LargeVehicles', 'NumberofLanes']].isnull().sum(axis=1)
    df['hour_x_lv'] = df['hour'] * df['LargeVehicles_encoded'].fillna(0)
    
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)
    df['temp_abs_from_25'] = np.abs(df['Temperature_filled'] - 25)
    
    def decode_geo(gh):
        if pd.isna(gh):
            return 0.0, 0.0
        return pgh.decode(gh)
    
    coords = df['geohash'].apply(decode_geo).tolist()
    df['lat'] = [c[0] for c in coords]
    df['lon'] = [c[1] for c in coords]
    
    return df

def get_r2_data(native_cat=False):
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    
    object_cols = [c for c in train_fe.columns if train_fe[c].dtype == 'object' and c not in ['demand', 'timestamp', 'geohash']]
    
    if not native_cat:
        for col in object_cols:
            le = LabelEncoder()
            combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)], axis=0)
            le.fit(combined)
            train_fe[col + '_le'] = le.transform(train_fe[col].astype(str))
            test_fe[col + '_le'] = le.transform(test_fe[col].astype(str))
    else:
        # Keep them as categorical types for CatBoost native support
        for col in object_cols + ['geohash']:
            train_fe[col] = train_fe[col].fillna('Missing').astype(str)
            test_fe[col] = test_fe[col].fillna('Missing').astype(str)
            
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    day48_mask = train['day'] == 48
    kmeans.fit(train_fe.loc[day48_mask, ['lat', 'lon']])
    train_fe['spatial_cluster_30'] = kmeans.predict(train_fe[['lat', 'lon']])
    test_fe['spatial_cluster_30'] = kmeans.predict(test_fe[['lat', 'lon']])
    
    # Best encoding from experiment: {best_enc}
    if "{best_enc}" == "Smoothed TE":
        global_mean = train_fe.loc[day48_mask, 'demand'].mean()
        alpha = 10
        geo_stats = train_fe[day48_mask].groupby('geohash')['demand'].agg(['mean', 'count'])
        geo_smoothed = ((global_mean * alpha) + (geo_stats['mean'] * geo_stats['count'])) / (alpha + geo_stats['count'])
        enc_dict = geo_smoothed.to_dict()
        fallback = global_mean
    elif "{best_enc}" == "Median TE":
        enc_dict = train_fe[day48_mask].groupby('geohash')['demand'].median().to_dict()
        fallback = train_fe.loc[day48_mask, 'demand'].median()
    else:
        enc_dict = train_fe[day48_mask].groupby('geohash')['demand'].mean().to_dict()
        fallback = train_fe.loc[day48_mask, 'demand'].mean()
        
    train_fe['encoded_demand'] = train_fe['geohash'].map(enc_dict).fillna(fallback)
    test_fe['encoded_demand'] = test_fe['geohash'].map(enc_dict).fillna(fallback)
    
    if not native_cat:
        drop_cols = ['Index', 'demand', 'timestamp', 'geohash', 'Temperature', 'day'] + object_cols
    else:
        drop_cols = ['Index', 'demand', 'timestamp', 'Temperature', 'day']
        
    drop_cols_existing = [c for c in drop_cols if c in train_fe.columns]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols_existing]
    
    X = train_fe[feature_cols].copy()
    y = train_fe['demand'].values
    X_test = test_fe[feature_cols].copy()
    
    # Imputation only for numericals
    num_cols = [c for c in feature_cols if X[c].dtype != 'object']
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])
    
    cat_features = [i for i, col in enumerate(feature_cols) if X[col].dtype == 'object'] if native_cat else []
    
    return X, y, X_test, feature_cols, cat_features
"""
with open('r2_fe_utils.py', 'w') as f:
    f.write(fe_utils_code)

print("\nWrote r2_fe_utils.py with optimized settings.")
print("DONE.")
