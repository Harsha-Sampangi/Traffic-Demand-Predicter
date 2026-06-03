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
    
    # Best encoding from experiment: Smoothed TE
    if "Smoothed TE" == "Smoothed TE":
        global_mean = train_fe.loc[day48_mask, 'demand'].mean()
        alpha = 10
        geo_stats = train_fe[day48_mask].groupby('geohash')['demand'].agg(['mean', 'count'])
        geo_smoothed = ((global_mean * alpha) + (geo_stats['mean'] * geo_stats['count'])) / (alpha + geo_stats['count'])
        enc_dict = geo_smoothed.to_dict()
        fallback = global_mean
    elif "Smoothed TE" == "Median TE":
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
