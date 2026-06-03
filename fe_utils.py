import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def parse_timestamp(df):
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = ts[0]
    df['minute'] = ts[1]
    return df

def engineer_features(df, is_train=True):
    df = df.copy()
    df = parse_timestamp(df)
    df['quarter_hour'] = df['hour'] * 4 + df['minute'] // 15
    df['time_bucket_30min'] = df['hour'] * 2 + df['minute'] // 30
    df['time_bucket_2hr'] = df['hour'] // 2
    df['time_bucket_3hr'] = df['hour'] // 3
    df['time_bucket_4hr'] = df['hour'] // 4
    df['time_bucket_6hr'] = df['hour'] // 6
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 18)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['quarter_hour_sin'] = np.sin(2 * np.pi * df['quarter_hour'] / 96)
    df['quarter_hour_cos'] = np.cos(2 * np.pi * df['quarter_hour'] / 96)
    df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    df['is_weekday'] = 1 - df['is_weekend']
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['week_number'] = df['day'] // 7
    for i in range(min(6, df['geohash'].str.len().min())):
        df[f'geohash_char_{i}'] = df['geohash'].str[i]
    df['geohash_prefix_3'] = df['geohash'].str[:3]
    df['geohash_prefix_4'] = df['geohash'].str[:4]
    df['geohash_prefix_5'] = df['geohash'].str[:5]
    roadtype_map = {'Residential': 0, 'Commercial': 1, 'Highway': 2}
    df['RoadType_encoded'] = df['RoadType'].map(roadtype_map)
    df['RoadType_missing'] = df['RoadType'].isnull().astype(int)
    lv_map = {'Not Allowed': 0, 'Allowed': 1}
    df['LargeVehicles_encoded'] = df['LargeVehicles'].map(lv_map)
    df['LargeVehicles_missing'] = df['LargeVehicles'].isnull().astype(int)
    landmarks_map = {'No': 0, 'Yes': 1}
    df['Landmarks_encoded'] = df['Landmarks'].map(landmarks_map)
    df['Landmarks_missing'] = df['Landmarks'].isnull().astype(int)
    weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}
    df['Weather_encoded'] = df['Weather'].map(weather_map)
    df['Weather_missing'] = df['Weather'].isnull().astype(int)
    df['Temperature_missing'] = df['Temperature'].isnull().astype(int)
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)
    df['temp_bin_5'] = pd.cut(df['Temperature_filled'], bins=5, labels=False)
    df['temp_bin_10'] = pd.cut(df['Temperature_filled'], bins=10, labels=False)
    df['is_cold'] = (df['Temperature_filled'] < 10).astype(int)
    df['is_hot'] = (df['Temperature_filled'] > 35).astype(int)
    df['temp_abs_from_25'] = np.abs(df['Temperature_filled'] - 25)
    df['NumberofLanes_missing'] = df['NumberofLanes'].isnull().astype(int)
    df['NumberofLanes_filled'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())
    df['is_multilane'] = (df['NumberofLanes_filled'] >= 3).astype(int)
    df['rush_x_roadtype'] = df['is_rush_hour'] * df['RoadType_encoded'].fillna(-1)
    df['night_x_temp'] = df['is_night'] * df['Temperature_filled']
    df['weekend_x_hour'] = df['is_weekend'] * df['hour']
    df['rush_x_lanes'] = df['is_rush_hour'] * df['NumberofLanes_filled']
    df['weather_x_hour'] = df['Weather_encoded'].fillna(-1) * df['hour']
    df['landmarks_x_rush'] = df['Landmarks_encoded'].fillna(0) * df['is_rush_hour']
    df['lv_x_lanes'] = df['LargeVehicles_encoded'].fillna(0) * df['NumberofLanes_filled']
    df['temp_x_weather'] = df['Temperature_filled'] * df['Weather_encoded'].fillna(0)
    df['hour_x_lv'] = df['hour'] * df['LargeVehicles_encoded'].fillna(0)
    missing_cols_check = ['RoadType', 'Temperature', 'Weather', 'Landmarks', 'LargeVehicles', 'NumberofLanes']
    existing_missing_cols = [c for c in missing_cols_check if c in df.columns]
    df['n_missing'] = df[existing_missing_cols].isnull().sum(axis=1)
    return df

def get_processed_data():
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    train_fe = engineer_features(train, is_train=True)
    test_fe = engineer_features(test, is_train=False)
    
    object_cols = [c for c in train_fe.columns if train_fe[c].dtype == 'object' and c != 'demand']
    
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)], axis=0)
        le.fit(combined)
        train_fe[col + '_le'] = le.transform(train_fe[col].astype(str))
        test_fe[col + '_le'] = le.transform(test_fe[col].astype(str))
        label_encoders[col] = le
        
    drop_cols = ['Index', 'demand', 'timestamp', 'geohash'] + object_cols
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
