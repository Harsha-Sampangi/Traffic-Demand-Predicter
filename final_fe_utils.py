import pandas as pd
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
    
    # We drop the 61 weak features identified in Step 5.
    # We will ONLY keep the high-value features.
    
    for i in range(min(6, df['geohash'].str.len().min())):
        df[f'geohash_char_{i}'] = df['geohash'].str[i]
    df['geohash_prefix_5'] = df['geohash'].str[:5]
    
    roadtype_map = {'Residential': 0, 'Commercial': 1, 'Highway': 2}
    df['RoadType_encoded'] = df['RoadType'].map(roadtype_map)
    df['RoadType_missing'] = df['RoadType'].isnull().astype(int)
    
    lv_map = {'Not Allowed': 0, 'Allowed': 1}
    df['LargeVehicles_encoded'] = df['LargeVehicles'].map(lv_map)
    
    landmarks_map = {'No': 0, 'Yes': 1}
    df['Landmarks_encoded'] = df['Landmarks'].map(landmarks_map)
    
    df['n_missing'] = df[['RoadType', 'Temperature', 'Weather', 'Landmarks', 'LargeVehicles', 'NumberofLanes']].isnull().sum(axis=1)
    
    df['hour_x_lv'] = df['hour'] * df['LargeVehicles_encoded'].fillna(0)
    
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)
    df['temp_abs_from_25'] = np.abs(df['Temperature_filled'] - 25)
    
    # Geohash Decode
    def decode_geo(gh):
        if pd.isna(gh):
            return 0.0, 0.0
        return pgh.decode(gh)
    
    coords = df['geohash'].apply(decode_geo).tolist()
    df['lat'] = [c[0] for c in coords]
    df['lon'] = [c[1] for c in coords]
    
    return df

def get_final_data():
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    
    # Label encode categories
    object_cols = [c for c in train_fe.columns if train_fe[c].dtype == 'object' and c not in ['demand', 'timestamp', 'geohash']]
    
    for col in object_cols:
        le = LabelEncoder()
        combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)], axis=0)
        le.fit(combined)
        train_fe[col + '_le'] = le.transform(train_fe[col].astype(str))
        test_fe[col + '_le'] = le.transform(test_fe[col].astype(str))
        
    # Spatial Clustering
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    # Fit only on train Day 48 to prevent leakage (technically fit on all train is fine if unsupervised, but let's be strict)
    day48_mask = train['day'] == 48
    kmeans.fit(train_fe.loc[day48_mask, ['lat', 'lon']])
    train_fe['spatial_cluster_30'] = kmeans.predict(train_fe[['lat', 'lon']])
    test_fe['spatial_cluster_30'] = kmeans.predict(test_fe[['lat', 'lon']])
    
    # Leak-safe target encoding (Mean demand per Geohash)
    # Fit ONLY on Day 48
    geo_demand = train_fe[day48_mask].groupby('geohash')['demand'].mean().to_dict()
    global_mean = train_fe.loc[day48_mask, 'demand'].mean()
    
    train_fe['geo_mean_demand'] = train_fe['geohash'].map(geo_demand).fillna(global_mean)
    test_fe['geo_mean_demand'] = test_fe['geohash'].map(geo_demand).fillna(global_mean)
    
    # Drop raw strings and original columns we encoded
    drop_cols = ['Index', 'demand', 'timestamp', 'geohash', 'Temperature', 'day'] + object_cols
    drop_cols_existing = [c for c in drop_cols if c in train_fe.columns]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols_existing]
    
    X = train_fe[feature_cols].copy()
    y = train_fe['demand'].values
    X_test = test_fe[feature_cols].copy()
    
    imputer = SimpleImputer(strategy='median')
    X_mat = imputer.fit_transform(X.values.astype(np.float32))
    X_test_mat = imputer.transform(X_test.values.astype(np.float32))
    
    X = pd.DataFrame(X_mat, columns=feature_cols)
    X_test = pd.DataFrame(X_test_mat, columns=feature_cols)
    
    return X, y, X_test, feature_cols
