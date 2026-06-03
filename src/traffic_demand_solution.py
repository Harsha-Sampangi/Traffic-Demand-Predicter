#!/usr/bin/env python3
"""
=============================================================================
Traffic Demand Prediction - Competition Solution
=============================================================================
Competition Objective: Predict traffic demand at various locations/times
Target Variable: demand (continuous)
Evaluation Metric: score = max(0, 100 * r2_score(actual, predicted))
Strategy: Multi-model ensemble with advanced feature engineering
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.INFO)

import shap
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
N_FOLDS = 5
OPTUNA_TRIALS = 15  # Per model
OUTPUT_DIR = 'output'
DATA_DIR = 'dataset'

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

print("=" * 80)
print("TRAFFIC DEMAND PREDICTION - COMPETITION SOLUTION")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"Test columns:  {list(test.columns)}")

TARGET = 'demand'

# ============================================================================
# 2. COMPREHENSIVE EDA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 2a. Data types
print("\n--- Data Types ---")
print(train.dtypes)

# 2b. Missing values
print("\n--- Missing Values (Train) ---")
missing_train = train.isnull().sum()
missing_pct_train = (train.isnull().sum() / len(train) * 100).round(2)
missing_df = pd.DataFrame({'count': missing_train, 'pct': missing_pct_train})
print(missing_df[missing_df['count'] > 0])

print("\n--- Missing Values (Test) ---")
missing_test = test.isnull().sum()
missing_pct_test = (test.isnull().sum() / len(test) * 100).round(2)
missing_df_test = pd.DataFrame({'count': missing_test, 'pct': missing_pct_test})
print(missing_df_test[missing_df_test['count'] > 0])

# 2c. Duplicate rows
print(f"\n--- Duplicate Rows ---")
print(f"Train duplicates: {train.duplicated().sum()}")
print(f"Test duplicates:  {test.duplicated().sum()}")

# 2d. Cardinality analysis
print("\n--- Cardinality Analysis ---")
for col in train.columns:
    print(f"  {col}: {train[col].nunique()} unique values")

# 2e. Target distribution
print(f"\n--- Target Distribution ---")
print(train[TARGET].describe())

# 2f. Numerical columns stats
print("\n--- Numerical Column Statistics ---")
print(train.describe())

# 2g. Categorical columns
cat_cols_raw = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for col in cat_cols_raw:
    if col in train.columns:
        print(f"\n  {col} value counts:")
        print(train[col].value_counts().head(10))

# Generate EDA visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Target distribution
axes[0, 0].hist(train[TARGET], bins=100, color='#4fc3f7', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Target Distribution (demand)')
axes[0, 0].set_xlabel('demand')
axes[0, 0].set_ylabel('Count')

# Log target distribution
axes[0, 1].hist(np.log1p(train[TARGET]), bins=100, color='#81c784', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Log(1 + demand) Distribution')
axes[0, 1].set_xlabel('log(1 + demand)')

# Temperature distribution
temp_valid = train['Temperature'].dropna()
axes[0, 2].hist(temp_valid, bins=50, color='#ffb74d', edgecolor='black', alpha=0.7)
axes[0, 2].set_title('Temperature Distribution')

# Day distribution
axes[1, 0].hist(train['day'], bins=train['day'].nunique(), color='#ce93d8', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Day Distribution')

# NumberofLanes distribution
axes[1, 1].hist(train['NumberofLanes'].dropna(), bins=10, color='#ef9a9a', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('NumberofLanes Distribution')

# Missing values heatmap
missing_cols = train.columns[train.isnull().any()].tolist()
if missing_cols:
    missing_matrix = train[missing_cols].isnull().astype(int).head(500)
    axes[1, 2].imshow(missing_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='none')
    axes[1, 2].set_yticks(range(len(missing_cols)))
    axes[1, 2].set_yticklabels(missing_cols)
    axes[1, 2].set_title('Missing Values Pattern (first 500 rows)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved EDA visualization to {OUTPUT_DIR}/eda_overview.png")

# Correlation analysis for numerical columns
num_cols_corr = ['day', 'NumberofLanes', 'Temperature', TARGET]
corr_data = train[num_cols_corr].dropna()
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = corr_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.3f')
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved correlation matrix to {OUTPUT_DIR}/correlation_matrix.png")

# Demand by Weather
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
if 'Weather' in train.columns:
    weather_demand = train.groupby('Weather')[TARGET].mean().sort_values(ascending=False)
    weather_demand.plot(kind='bar', ax=axes[0], color='#4fc3f7')
    axes[0].set_title('Mean Demand by Weather')
    axes[0].tick_params(axis='x', rotation=45)

if 'RoadType' in train.columns:
    rt_demand = train.groupby('RoadType')[TARGET].mean().sort_values(ascending=False)
    rt_demand.plot(kind='bar', ax=axes[1], color='#81c784')
    axes[1].set_title('Mean Demand by RoadType')
    axes[1].tick_params(axis='x', rotation=45)

if 'LargeVehicles' in train.columns:
    lv_demand = train.groupby('LargeVehicles')[TARGET].mean().sort_values(ascending=False)
    lv_demand.plot(kind='bar', ax=axes[2], color='#ffb74d')
    axes[2].set_title('Mean Demand by LargeVehicles')
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'demand_by_category.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved category analysis to {OUTPUT_DIR}/demand_by_category.png")

# Outlier detection
print("\n--- Outlier Detection (IQR Method) ---")
Q1 = train[TARGET].quantile(0.25)
Q3 = train[TARGET].quantile(0.75)
IQR = Q3 - Q1
outliers = ((train[TARGET] < Q1 - 1.5 * IQR) | (train[TARGET] > Q3 + 1.5 * IQR)).sum()
print(f"Target outliers (IQR): {outliers} ({outliers/len(train)*100:.2f}%)")
print(f"Target range: [{train[TARGET].min():.6f}, {train[TARGET].max():.6f}]")

# Leakage detection
print("\n--- Leakage Detection ---")
print("Checking if Index is sequential and non-informative...")
print(f"Index range train: [{train['Index'].min()}, {train['Index'].max()}]")
print(f"Index range test:  [{test['Index'].min()}, {test['Index'].max()}]")
idx_corr = train[['Index', TARGET]].corr().iloc[0, 1]
print(f"Index-Target correlation: {idx_corr:.6f}")
if abs(idx_corr) < 0.05:
    print("=> Index appears non-informative (no leakage)")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

def parse_timestamp(df):
    """Parse timestamp column (format: H:M) into components."""
    ts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = ts[0]
    df['minute'] = ts[1]
    return df

def engineer_features(df, is_train=True, target_encodings=None):
    """
    Comprehensive feature engineering pipeline.
    Returns the dataframe with new features and any fitted encodings.
    """
    df = df.copy()

    # ---- TIMESTAMP FEATURES ----
    df = parse_timestamp(df)

    # Quarter-hour slot (0-95 for 96 15-min intervals)
    df['quarter_hour'] = df['hour'] * 4 + df['minute'] // 15

    # Time bucket features
    df['time_bucket_30min'] = df['hour'] * 2 + df['minute'] // 30
    df['time_bucket_2hr'] = df['hour'] // 2
    df['time_bucket_3hr'] = df['hour'] // 3
    df['time_bucket_4hr'] = df['hour'] // 4
    df['time_bucket_6hr'] = df['hour'] // 6

    # Rush hour indicators
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # Night indicator
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(int)

    # Business hours
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # Peak traffic indicators
    df['is_peak_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 9)).astype(int)
    df['is_peak_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 18)).astype(int)

    # Cyclic features using sine/cosine encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['quarter_hour_sin'] = np.sin(2 * np.pi * df['quarter_hour'] / 96)
    df['quarter_hour_cos'] = np.cos(2 * np.pi * df['quarter_hour'] / 96)

    # Minutes since midnight (continuous time feature)
    df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']

    # ---- DAY FEATURES ----
    # Day of week (assuming day is sequential, modulo 7)
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    df['is_weekday'] = 1 - df['is_weekend']

    # Day cyclic
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Week number from day
    df['week_number'] = df['day'] // 7

    # ---- GEOHASH FEATURES ----
    # Decode geohash characters as categorical levels
    for i in range(min(6, df['geohash'].str.len().min())):
        df[f'geohash_char_{i}'] = df['geohash'].str[i]

    # Geohash prefix features (spatial hierarchy)
    df['geohash_prefix_3'] = df['geohash'].str[:3]
    df['geohash_prefix_4'] = df['geohash'].str[:4]
    df['geohash_prefix_5'] = df['geohash'].str[:5]

    # ---- CATEGORICAL ENCODING ----
    # RoadType
    roadtype_map = {'Residential': 0, 'Commercial': 1, 'Highway': 2}
    df['RoadType_encoded'] = df['RoadType'].map(roadtype_map)
    df['RoadType_missing'] = df['RoadType'].isnull().astype(int)

    # LargeVehicles
    lv_map = {'Not Allowed': 0, 'Allowed': 1}
    df['LargeVehicles_encoded'] = df['LargeVehicles'].map(lv_map)
    df['LargeVehicles_missing'] = df['LargeVehicles'].isnull().astype(int)

    # Landmarks
    landmarks_map = {'No': 0, 'Yes': 1}
    df['Landmarks_encoded'] = df['Landmarks'].map(landmarks_map)
    df['Landmarks_missing'] = df['Landmarks'].isnull().astype(int)

    # Weather
    weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}
    df['Weather_encoded'] = df['Weather'].map(weather_map)
    df['Weather_missing'] = df['Weather'].isnull().astype(int)

    # ---- TEMPERATURE FEATURES ----
    df['Temperature_missing'] = df['Temperature'].isnull().astype(int)
    # Fill missing temperature with median (will be overridden per fold for target encoding)
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)

    # Temperature bins
    df['temp_bin_5'] = pd.cut(df['Temperature_filled'], bins=5, labels=False)
    df['temp_bin_10'] = pd.cut(df['Temperature_filled'], bins=10, labels=False)

    # Temperature extremes
    df['is_cold'] = (df['Temperature_filled'] < 10).astype(int)
    df['is_hot'] = (df['Temperature_filled'] > 35).astype(int)
    df['temp_abs_from_25'] = np.abs(df['Temperature_filled'] - 25)

    # ---- NumberofLanes FEATURES ----
    df['NumberofLanes_missing'] = df['NumberofLanes'].isnull().astype(int)
    df['NumberofLanes_filled'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())
    df['is_multilane'] = (df['NumberofLanes_filled'] >= 3).astype(int)

    # ---- INTERACTION FEATURES ----
    # Rush hour × Road type
    df['rush_x_roadtype'] = df['is_rush_hour'] * df['RoadType_encoded'].fillna(-1)

    # Night × Temperature
    df['night_x_temp'] = df['is_night'] * df['Temperature_filled']

    # Weekend × hour
    df['weekend_x_hour'] = df['is_weekend'] * df['hour']

    # Rush hour × lanes
    df['rush_x_lanes'] = df['is_rush_hour'] * df['NumberofLanes_filled']

    # Weather × hour
    df['weather_x_hour'] = df['Weather_encoded'].fillna(-1) * df['hour']

    # Landmarks × rush hour
    df['landmarks_x_rush'] = df['Landmarks_encoded'].fillna(0) * df['is_rush_hour']

    # LargeVehicles × lanes
    df['lv_x_lanes'] = df['LargeVehicles_encoded'].fillna(0) * df['NumberofLanes_filled']

    # Temperature × Weather interaction
    df['temp_x_weather'] = df['Temperature_filled'] * df['Weather_encoded'].fillna(0)

    # Hour × LargeVehicles
    df['hour_x_lv'] = df['hour'] * df['LargeVehicles_encoded'].fillna(0)

    # Number of missing values per row (data quality indicator)
    missing_cols_check = ['RoadType', 'Temperature', 'Weather', 'Landmarks', 'LargeVehicles', 'NumberofLanes']
    existing_missing_cols = [c for c in missing_cols_check if c in df.columns]
    df['n_missing'] = df[existing_missing_cols].isnull().sum(axis=1)

    return df


# Apply feature engineering
print("Applying feature engineering to train...")
train_fe = engineer_features(train, is_train=True)
print(f"Train features after FE: {train_fe.shape[1]}")

print("Applying feature engineering to test...")
test_fe = engineer_features(test, is_train=False)
print(f"Test features after FE: {test_fe.shape[1]}")

# ============================================================================
# 4. LABEL ENCODING FOR REMAINING CATEGORICALS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: LABEL ENCODING")
print("=" * 80)

# Identify remaining object columns that need encoding
object_cols = [c for c in train_fe.columns if train_fe[c].dtype == 'object' and c != TARGET]
print(f"Object columns to encode: {object_cols}")

label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    # Combine train and test for consistent encoding
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)], axis=0)
    le.fit(combined)
    train_fe[col + '_le'] = le.transform(train_fe[col].astype(str))
    test_fe[col + '_le'] = le.transform(test_fe[col].astype(str))
    label_encoders[col] = le

# ============================================================================
# 5. PREPARE FEATURES FOR MODELING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: PREPARE FEATURES FOR MODELING")
print("=" * 80)

# Drop columns not needed for modeling
drop_cols = ['Index', TARGET, 'timestamp', 'geohash'] + object_cols
# Also drop the original categorical columns (already encoded)
drop_cols_existing = [c for c in drop_cols if c in train_fe.columns]

feature_cols = [c for c in train_fe.columns if c not in drop_cols_existing]
print(f"Number of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

X = train_fe[feature_cols].values.astype(np.float32)
y = train_fe[TARGET].values.astype(np.float64)
X_test = test_fe[feature_cols].values.astype(np.float32)
test_index = test['Index'].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# Handle any remaining NaN in features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# ============================================================================
# 6. TARGET ENCODING (Inside CV folds to prevent leakage)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: TARGET ENCODING (Leak-safe)")
print("=" * 80)

# We'll add target encodings as features but only fit on train folds
# Columns to target-encode: geohash, geohash_prefix_3, geohash_prefix_4, day_of_week, hour, quarter_hour
te_source_cols = ['geohash', 'geohash_prefix_3', 'geohash_prefix_4', 'geohash_prefix_5']
te_source_cols_extra = ['hour', 'quarter_hour', 'day_of_week', 'RoadType', 'Weather']

# Build target encoding using full train with regularized mean
def build_target_encoding(train_df, test_df, col, target, alpha=10):
    """Build regularized target encoding for a column."""
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    agg['te'] = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
    te_map = agg['te'].to_dict()

    train_te = train_df[col].map(te_map).fillna(global_mean)
    test_te = test_df[col].map(te_map).fillna(global_mean)
    return train_te.values, test_te.values

# For the main model features, we'll compute target encodings inside CV
# For now, let's prepare the raw columns needed
te_cols_in_train = []
te_cols_in_test = []

for col in te_source_cols + te_source_cols_extra:
    if col in train_fe.columns:
        col_name = f'te_{col}'
        train_te_val, test_te_val = build_target_encoding(train_fe, test_fe, col, TARGET)
        train_fe[col_name] = train_te_val
        test_fe[col_name] = test_te_val
        te_cols_in_train.append(col_name)

# Also add frequency encoding
for col in te_source_cols + ['RoadType', 'Weather', 'LargeVehicles']:
    if col in train_fe.columns:
        freq = train_fe[col].value_counts(normalize=True).to_dict()
        train_fe[f'freq_{col}'] = train_fe[col].map(freq).fillna(0)
        test_fe[f'freq_{col}'] = test_fe[col].map(freq).fillna(0)

# Location-level aggregate statistics
print("Computing location-level aggregate statistics...")
for col in ['geohash', 'geohash_prefix_4']:
    if col in train_fe.columns:
        agg_stats = train_fe.groupby(col)['Temperature_filled'].agg(['mean', 'std', 'min', 'max']).reset_index()
        agg_stats.columns = [col, f'{col}_temp_mean', f'{col}_temp_std', f'{col}_temp_min', f'{col}_temp_max']
        train_fe = train_fe.merge(agg_stats, on=col, how='left')
        test_fe = test_fe.merge(agg_stats, on=col, how='left')

        # Count per location
        count_stats = train_fe.groupby(col).size().reset_index(name=f'{col}_count')
        train_fe = train_fe.merge(count_stats, on=col, how='left')
        test_fe = test_fe.merge(count_stats, on=col, how='left')

# Rebuild feature columns
drop_cols_final = ['Index', TARGET, 'timestamp', 'geohash'] + object_cols
feature_cols_final = [c for c in train_fe.columns
                      if c not in drop_cols_final
                      and train_fe[c].dtype in ['int64', 'float64', 'float32', 'int32', 'int8', 'uint8']]

# Remove duplicates
feature_cols_final = list(dict.fromkeys(feature_cols_final))

print(f"Final number of features: {len(feature_cols_final)}")

X = train_fe[feature_cols_final].values.astype(np.float32)
y = train_fe[TARGET].values.astype(np.float64)
X_test = test_fe[feature_cols_final].values.astype(np.float32)

# Re-impute
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

print(f"X shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test.shape}")

# ============================================================================
# 7. VALIDATION STRATEGY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: VALIDATION STRATEGY")
print("=" * 80)

# Use stratified KFold based on target bins
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y_bins))

print(f"Using {N_FOLDS}-Fold Stratified Cross-Validation (target-binned)")
for i, (train_idx, val_idx) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}")

# ============================================================================
# 8. BASELINE MODELS (Quick evaluation before HPO)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: BASELINE MODEL EVALUATION")
print("=" * 80)

def evaluate_model_cv(model_fn, X, y, folds, X_test, model_name="Model"):
    """Evaluate a model using cross-validation and return OOF predictions and test predictions."""
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()

        if isinstance(model, cb.CatBoostRegressor):
            model.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     verbose=0,
                     early_stopping_rounds=100)
        elif isinstance(model, lgb.LGBMRegressor):
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        elif isinstance(model, xgb.XGBRegressor):
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=0)
        else:
            model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / len(folds)

        r2 = r2_score(y_val, val_pred)
        score = max(0, 100 * r2)
        fold_scores.append(score)
        print(f"  {model_name} Fold {fold_idx+1}: R²={r2:.6f}, Score={score:.4f}")

    overall_r2 = r2_score(y, oof_preds)
    overall_score = max(0, 100 * overall_r2)
    print(f"  {model_name} Mean Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  {model_name} Overall OOF R²: {overall_r2:.6f}, Score: {overall_score:.4f}")

    return oof_preds, test_preds, fold_scores, overall_score

# Quick baseline with default params
print("\n--- Baseline CatBoost ---")
cb_baseline_oof, cb_baseline_test, cb_baseline_scores, cb_baseline_overall = evaluate_model_cv(
    lambda: cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6,
                                  random_seed=SEED, verbose=0,
                                  early_stopping_rounds=100),
    X, y, folds, X_test, "CatBoost-Baseline"
)

print("\n--- Baseline LightGBM ---")
lgb_baseline_oof, lgb_baseline_test, lgb_baseline_scores, lgb_baseline_overall = evaluate_model_cv(
    lambda: lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6,
                               random_state=SEED, verbose=-1, n_jobs=-1),
    X, y, folds, X_test, "LightGBM-Baseline"
)

print("\n--- Baseline XGBoost ---")
xgb_baseline_oof, xgb_baseline_test, xgb_baseline_scores, xgb_baseline_overall = evaluate_model_cv(
    lambda: xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6,
                              random_state=SEED, verbosity=0, n_jobs=-1,
                              early_stopping_rounds=100),
    X, y, folds, X_test, "XGBoost-Baseline"
)

# ============================================================================
# 9. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: HYPERPARAMETER OPTIMIZATION (Optuna)")
print("=" * 80)

# Use a single fold for faster HPO, then retrain on all folds
hpo_fold_idx = 0
hpo_train_idx, hpo_val_idx = folds[hpo_fold_idx]
X_hpo_train, X_hpo_val = X[hpo_train_idx], X[hpo_val_idx]
y_hpo_train, y_hpo_val = y[hpo_train_idx], y[hpo_val_idx]

# --- CatBoost HPO ---
print("\n--- CatBoost Optuna Optimization ---")

def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 100,
    }

    model = cb.CatBoostRegressor(**params)
    model.fit(X_hpo_train, y_hpo_train, eval_set=(X_hpo_val, y_hpo_val), verbose=0)
    pred = model.predict(X_hpo_val)
    return r2_score(y_hpo_val, pred)

cb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
cb_study.optimize(catboost_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
cb_best_params = cb_study.best_params
print(f"CatBoost best R²: {cb_study.best_value:.6f}")
print(f"CatBoost best params: {cb_best_params}")

# --- LightGBM HPO ---
print("\n--- LightGBM Optuna Optimization ---")

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_hpo_train, y_hpo_train,
             eval_set=[(X_hpo_val, y_hpo_val)],
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = model.predict(X_hpo_val)
    return r2_score(y_hpo_val, pred)

lgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
lgb_study.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
lgb_best_params = lgb_study.best_params
print(f"LightGBM best R²: {lgb_study.best_value:.6f}")
print(f"LightGBM best params: {lgb_best_params}")

# --- XGBoost HPO ---
print("\n--- XGBoost Optuna Optimization ---")

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': SEED,
        'verbosity': 0,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_hpo_train, y_hpo_train,
             eval_set=[(X_hpo_val, y_hpo_val)],
             verbose=0)
    pred = model.predict(X_hpo_val)
    return r2_score(y_hpo_val, pred)

xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
xgb_best_params = xgb_study.best_params
print(f"XGBoost best R²: {xgb_study.best_value:.6f}")
print(f"XGBoost best params: {xgb_best_params}")

# --- HistGradientBoosting HPO ---
print("\n--- HistGradientBoosting Optuna Optimization ---")

def hgb_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 255),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255),
        'random_state': SEED,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
    }

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_hpo_train, y_hpo_train)
    pred = model.predict(X_hpo_val)
    return r2_score(y_hpo_val, pred)

hgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
hgb_study.optimize(hgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
hgb_best_params = hgb_study.best_params
print(f"HistGBR best R²: {hgb_study.best_value:.6f}")
print(f"HistGBR best params: {hgb_best_params}")

# --- ExtraTrees HPO ---
print("\n--- ExtraTrees Optuna Optimization ---")

def et_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': SEED,
        'n_jobs': -1,
    }

    model = ExtraTreesRegressor(**params)
    model.fit(X_hpo_train, y_hpo_train)
    pred = model.predict(X_hpo_val)
    return r2_score(y_hpo_val, pred)

et_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
et_study.optimize(et_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
et_best_params = et_study.best_params
print(f"ExtraTrees best R²: {et_study.best_value:.6f}")
print(f"ExtraTrees best params: {et_best_params}")

# ============================================================================
# 10. RETRAIN WITH BEST PARAMS (Full CV)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: RETRAINING WITH OPTIMIZED HYPERPARAMETERS")
print("=" * 80)

# CatBoost with best params
print("\n--- Optimized CatBoost ---")
cb_params_final = {k: v for k, v in cb_best_params.items()}
cb_params_final['random_seed'] = SEED
cb_params_final['verbose'] = 0
cb_params_final['early_stopping_rounds'] = 100
cb_oof, cb_test, cb_fold_scores, cb_overall = evaluate_model_cv(
    lambda: cb.CatBoostRegressor(**cb_params_final),
    X, y, folds, X_test, "CatBoost-Optimized"
)

# LightGBM with best params
print("\n--- Optimized LightGBM ---")
lgb_params_final = {k: v for k, v in lgb_best_params.items()}
lgb_params_final['random_state'] = SEED
lgb_params_final['verbose'] = -1
lgb_params_final['n_jobs'] = -1
lgb_oof, lgb_test, lgb_fold_scores, lgb_overall = evaluate_model_cv(
    lambda: lgb.LGBMRegressor(**lgb_params_final),
    X, y, folds, X_test, "LightGBM-Optimized"
)

# XGBoost with best params
print("\n--- Optimized XGBoost ---")
xgb_params_final = {k: v for k, v in xgb_best_params.items()}
xgb_params_final['random_state'] = SEED
xgb_params_final['verbosity'] = 0
xgb_params_final['n_jobs'] = -1
xgb_params_final['early_stopping_rounds'] = 100
xgb_oof, xgb_test, xgb_fold_scores, xgb_overall = evaluate_model_cv(
    lambda: xgb.XGBRegressor(**xgb_params_final),
    X, y, folds, X_test, "XGBoost-Optimized"
)

# HistGradientBoosting with best params
print("\n--- Optimized HistGradientBoosting ---")
hgb_params_final = {k: v for k, v in hgb_best_params.items()}
hgb_params_final['random_state'] = SEED
hgb_params_final['early_stopping'] = True
hgb_params_final['validation_fraction'] = 0.1
hgb_params_final['n_iter_no_change'] = 50
hgb_oof, hgb_test, hgb_fold_scores, hgb_overall = evaluate_model_cv(
    lambda: HistGradientBoostingRegressor(**hgb_params_final),
    X, y, folds, X_test, "HistGBR-Optimized"
)

# ExtraTrees with best params
print("\n--- Optimized ExtraTrees ---")
et_params_final = {k: v for k, v in et_best_params.items()}
et_params_final['random_state'] = SEED
et_params_final['n_jobs'] = -1
et_oof, et_test, et_fold_scores, et_overall = evaluate_model_cv(
    lambda: ExtraTreesRegressor(**et_params_final),
    X, y, folds, X_test, "ExtraTrees-Optimized"
)

# ============================================================================
# 11. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: MODEL COMPARISON")
print("=" * 80)

model_results = {
    'CatBoost': {'oof': cb_oof, 'test': cb_test, 'scores': cb_fold_scores, 'overall': cb_overall},
    'LightGBM': {'oof': lgb_oof, 'test': lgb_test, 'scores': lgb_fold_scores, 'overall': lgb_overall},
    'XGBoost': {'oof': xgb_oof, 'test': xgb_test, 'scores': xgb_fold_scores, 'overall': xgb_overall},
    'HistGBR': {'oof': hgb_oof, 'test': hgb_test, 'scores': hgb_fold_scores, 'overall': hgb_overall},
    'ExtraTrees': {'oof': et_oof, 'test': et_test, 'scores': et_fold_scores, 'overall': et_overall},
}

comparison_data = []
for name, result in model_results.items():
    comparison_data.append({
        'Model': name,
        'Mean_Score': np.mean(result['scores']),
        'Std_Score': np.std(result['scores']),
        'Overall_Score': result['overall'],
        'Fold_1': result['scores'][0],
        'Fold_2': result['scores'][1],
        'Fold_3': result['scores'][2],
        'Fold_4': result['scores'][3],
        'Fold_5': result['scores'][4],
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('Overall_Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
print(f"\nSaved model comparison to {OUTPUT_DIR}/model_comparison.csv")

# ============================================================================
# 12. ENSEMBLING - Weighted Blending
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: ENSEMBLE - OPTIMAL WEIGHT SEARCH")
print("=" * 80)

# Level 1: CatBoost, LightGBM, XGBoost (top 3 boosting models)
# Search optimal weights using grid search on OOF predictions

from itertools import product

def search_optimal_weights(oof_preds_list, y_true, model_names, n_steps=21):
    """Search for optimal blending weights using grid search."""
    best_score = -np.inf
    best_weights = None

    weights_range = np.linspace(0, 1, n_steps)
    n_models = len(oof_preds_list)

    if n_models == 2:
        for w1 in weights_range:
            w2 = 1 - w1
            blend = w1 * oof_preds_list[0] + w2 * oof_preds_list[1]
            score = max(0, 100 * r2_score(y_true, blend))
            if score > best_score:
                best_score = score
                best_weights = [w1, w2]

    elif n_models == 3:
        for w1 in weights_range:
            for w2 in weights_range:
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                blend = w1 * oof_preds_list[0] + w2 * oof_preds_list[1] + w3 * oof_preds_list[2]
                score = max(0, 100 * r2_score(y_true, blend))
                if score > best_score:
                    best_score = score
                    best_weights = [w1, w2, w3]

    elif n_models == 4:
        for w1 in weights_range:
            for w2 in weights_range:
                for w3 in weights_range:
                    w4 = 1 - w1 - w2 - w3
                    if w4 < 0:
                        continue
                    blend = (w1 * oof_preds_list[0] + w2 * oof_preds_list[1] +
                             w3 * oof_preds_list[2] + w4 * oof_preds_list[3])
                    score = max(0, 100 * r2_score(y_true, blend))
                    if score > best_score:
                        best_score = score
                        best_weights = [w1, w2, w3, w4]

    elif n_models == 5:
        # For 5 models, use coarser grid then refine
        coarse_range = np.linspace(0, 1, 11)
        for w1 in coarse_range:
            for w2 in coarse_range:
                for w3 in coarse_range:
                    for w4 in coarse_range:
                        w5 = 1 - w1 - w2 - w3 - w4
                        if w5 < 0:
                            continue
                        blend = (w1 * oof_preds_list[0] + w2 * oof_preds_list[1] +
                                 w3 * oof_preds_list[2] + w4 * oof_preds_list[3] +
                                 w5 * oof_preds_list[4])
                        score = max(0, 100 * r2_score(y_true, blend))
                        if score > best_score:
                            best_score = score
                            best_weights = [w1, w2, w3, w4, w5]

    for name, w in zip(model_names, best_weights):
        print(f"  {name}: {w:.4f}")
    print(f"  Ensemble Score: {best_score:.4f}")
    return best_weights, best_score


# 3-model ensemble (CatBoost, LightGBM, XGBoost)
print("\n--- 3-Model Ensemble (CB + LGB + XGB) ---")
oof_3 = [cb_oof, lgb_oof, xgb_oof]
test_3 = [cb_test, lgb_test, xgb_test]
names_3 = ['CatBoost', 'LightGBM', 'XGBoost']
weights_3, score_3 = search_optimal_weights(oof_3, y, names_3, n_steps=21)

# 4-model ensemble (adding HistGBR)
print("\n--- 4-Model Ensemble (CB + LGB + XGB + HGB) ---")
oof_4 = [cb_oof, lgb_oof, xgb_oof, hgb_oof]
test_4 = [cb_test, lgb_test, xgb_test, hgb_test]
names_4 = ['CatBoost', 'LightGBM', 'XGBoost', 'HistGBR']
weights_4, score_4 = search_optimal_weights(oof_4, y, names_4, n_steps=11)

# 5-model ensemble (all models)
print("\n--- 5-Model Ensemble (All models) ---")
oof_5 = [cb_oof, lgb_oof, xgb_oof, hgb_oof, et_oof]
test_5 = [cb_test, lgb_test, xgb_test, hgb_test, et_test]
names_5 = ['CatBoost', 'LightGBM', 'XGBoost', 'HistGBR', 'ExtraTrees']
weights_5, score_5 = search_optimal_weights(oof_5, y, names_5, n_steps=11)

# Simple average ensemble
avg_oof_3 = np.mean(oof_3, axis=0)
avg_test_3 = np.mean(test_3, axis=0)
avg_score_3 = max(0, 100 * r2_score(y, avg_oof_3))
print(f"\n--- Simple Average (3 models) Score: {avg_score_3:.4f} ---")

avg_oof_5 = np.mean(oof_5, axis=0)
avg_test_5 = np.mean(test_5, axis=0)
avg_score_5 = max(0, 100 * r2_score(y, avg_oof_5))
print(f"--- Simple Average (5 models) Score: {avg_score_5:.4f} ---")

# Select best ensemble
ensemble_options = {
    '3-model weighted': (score_3, weights_3, oof_3, test_3, names_3),
    '4-model weighted': (score_4, weights_4, oof_4, test_4, names_4),
    '5-model weighted': (score_5, weights_5, oof_5, test_5, names_5),
    '3-model average': (avg_score_3, [1/3, 1/3, 1/3], oof_3, test_3, names_3),
    '5-model average': (avg_score_5, [0.2]*5, oof_5, test_5, names_5),
}

# Also compare individual model scores
all_scores = {}
for name, res in model_results.items():
    all_scores[f'Individual-{name}'] = res['overall']
for name, (score, _, _, _, _) in ensemble_options.items():
    all_scores[f'Ensemble-{name}'] = score

print("\n--- ALL SCORES COMPARISON ---")
for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
    print(f"  {name}: {score:.4f}")

# Pick the best overall solution
best_solution_name = max(all_scores, key=all_scores.get)
best_solution_score = all_scores[best_solution_name]
print(f"\n*** BEST SOLUTION: {best_solution_name} (Score: {best_solution_score:.4f}) ***")

# Generate final predictions
if best_solution_name.startswith('Ensemble-'):
    ensemble_key = best_solution_name.replace('Ensemble-', '')
    _, best_weights, best_oofs, best_tests, best_names = ensemble_options[ensemble_key]
    final_test_preds = sum(w * t for w, t in zip(best_weights, best_tests))
    final_oof_preds = sum(w * o for w, o in zip(best_weights, best_oofs))
else:
    model_key = best_solution_name.replace('Individual-', '')
    final_test_preds = model_results[model_key]['test']
    final_oof_preds = model_results[model_key]['oof']

final_r2 = r2_score(y, final_oof_preds)
final_score = max(0, 100 * final_r2)
print(f"Final OOF R²: {final_r2:.6f}")
print(f"Final Competition Score: {final_score:.4f}")

# ============================================================================
# 13. FEATURE IMPORTANCE & SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: FEATURE IMPORTANCE & SHAP ANALYSIS")
print("=" * 80)

# Train a single LightGBM on full data for feature importance
lgb_fi_model = lgb.LGBMRegressor(**lgb_params_final)
lgb_fi_model.fit(X, y)

# Feature importance (gain-based)
fi_gain = lgb_fi_model.feature_importances_
fi_df = pd.DataFrame({
    'feature': feature_cols_final,
    'importance_gain': fi_gain
}).sort_values('importance_gain', ascending=False)

print("\nTop 30 Features (LightGBM Gain):")
print(fi_df.head(30).to_string(index=False))
fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# SHAP analysis (on a sample for speed)
print("\nComputing SHAP values...")
sample_size = min(5000, len(X))
X_sample = X[:sample_size]

try:
    explainer = shap.TreeExplainer(lgb_fi_model)
    shap_values = explainer.shap_values(X_sample)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample,
                     feature_names=feature_cols_final,
                     show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {OUTPUT_DIR}/shap_summary.png")

    # SHAP importance bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample,
                     feature_names=feature_cols_final,
                     plot_type='bar', show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_importance_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP importance bar plot to {OUTPUT_DIR}/shap_importance_bar.png")

    # SHAP-based feature importance
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_fi_df = pd.DataFrame({
        'feature': feature_cols_final,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    print("\nTop 20 Features (SHAP):")
    print(shap_fi_df.head(20).to_string(index=False))

except Exception as e:
    print(f"SHAP analysis failed: {e}")
    shap_fi_df = fi_df.copy()

# Permutation importance
print("\nComputing Permutation Importance...")
try:
    perm_imp = permutation_importance(lgb_fi_model, X_hpo_val, y_hpo_val,
                                       n_repeats=5, random_state=SEED, n_jobs=-1,
                                       scoring='r2')
    perm_fi_df = pd.DataFrame({
        'feature': feature_cols_final,
        'perm_importance_mean': perm_imp.importances_mean,
        'perm_importance_std': perm_imp.importances_std
    }).sort_values('perm_importance_mean', ascending=False)
    print("\nTop 20 Features (Permutation):")
    print(perm_fi_df.head(20).to_string(index=False))

    # Identify harmful features (negative permutation importance)
    harmful_features = perm_fi_df[perm_fi_df['perm_importance_mean'] < -0.001]['feature'].tolist()
    if harmful_features:
        print(f"\nPotentially harmful features: {harmful_features}")
    else:
        print("\nNo clearly harmful features detected.")

except Exception as e:
    print(f"Permutation importance failed: {e}")
    harmful_features = []

# Feature importance visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# LightGBM gain importance
top_n = 30
top_fi = fi_df.head(top_n)
axes[0].barh(range(top_n), top_fi['importance_gain'].values[::-1], color='#4fc3f7')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_fi['feature'].values[::-1], fontsize=8)
axes[0].set_title(f'Top {top_n} Features (LightGBM Gain)')
axes[0].set_xlabel('Importance (Gain)')

# SHAP importance
top_shap = shap_fi_df.head(top_n)
axes[1].barh(range(top_n), top_shap['shap_importance'].values[::-1], color='#81c784')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_shap['feature'].values[::-1], fontsize=8)
axes[1].set_title(f'Top {top_n} Features (SHAP)')
axes[1].set_xlabel('Mean |SHAP Value|')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved feature importance comparison to {OUTPUT_DIR}/feature_importance_comparison.png")

# ============================================================================
# 14. FEATURE SELECTION & RETRAIN (if harmful features found)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 14: FEATURE SELECTION & RETRAIN")
print("=" * 80)

if harmful_features and len(harmful_features) > 0:
    print(f"Removing {len(harmful_features)} harmful features and retraining...")
    selected_features = [f for f in feature_cols_final if f not in harmful_features]
    print(f"Features reduced from {len(feature_cols_final)} to {len(selected_features)}")

    X_sel = train_fe[selected_features].values.astype(np.float32)
    X_test_sel = test_fe[selected_features].values.astype(np.float32)
    X_sel = imputer.fit_transform(X_sel)
    X_test_sel = imputer.transform(X_test_sel)

    # Quick retrain of top models
    print("\n--- Retrained CatBoost ---")
    cb_sel_oof, cb_sel_test, cb_sel_scores, cb_sel_overall = evaluate_model_cv(
        lambda: cb.CatBoostRegressor(**cb_params_final),
        X_sel, y, folds, X_test_sel, "CatBoost-Selected"
    )

    print("\n--- Retrained LightGBM ---")
    lgb_sel_oof, lgb_sel_test, lgb_sel_scores, lgb_sel_overall = evaluate_model_cv(
        lambda: lgb.LGBMRegressor(**lgb_params_final),
        X_sel, y, folds, X_test_sel, "LightGBM-Selected"
    )

    print("\n--- Retrained XGBoost ---")
    xgb_sel_oof, xgb_sel_test, xgb_sel_scores, xgb_sel_overall = evaluate_model_cv(
        lambda: xgb.XGBRegressor(**xgb_params_final),
        X_sel, y, folds, X_test_sel, "XGBoost-Selected"
    )

    # Re-ensemble
    print("\n--- Re-ensemble with selected features ---")
    sel_oof_3 = [cb_sel_oof, lgb_sel_oof, xgb_sel_oof]
    sel_test_3 = [cb_sel_test, lgb_sel_test, xgb_sel_test]
    sel_names_3 = ['CatBoost-Sel', 'LightGBM-Sel', 'XGBoost-Sel']
    sel_weights_3, sel_score_3 = search_optimal_weights(sel_oof_3, y, sel_names_3, n_steps=21)

    if sel_score_3 > final_score:
        print(f"\nFeature selection IMPROVED score: {final_score:.4f} -> {sel_score_3:.4f}")
        final_test_preds = sum(w * t for w, t in zip(sel_weights_3, sel_test_3))
        final_oof_preds = sum(w * o for w, o in zip(sel_weights_3, sel_oof_3))
        final_score = sel_score_3
        best_solution_name = "Ensemble-3model-FeatureSelected"
    else:
        print(f"\nFeature selection did NOT improve score: {final_score:.4f} vs {sel_score_3:.4f}")
        print("Keeping original features.")
else:
    print("No harmful features to remove. Skipping retraining.")

# ============================================================================
# 15. SAVE BEST MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 15: SAVE BEST MODEL")
print("=" * 80)

# Save the best single model (LightGBM or CatBoost based on performance)
best_single = comparison_df.iloc[0]['Model']
print(f"Best single model: {best_single}")

if best_single == 'CatBoost':
    best_model = cb.CatBoostRegressor(**cb_params_final)
    best_model.fit(X, y, verbose=0)
elif best_single == 'LightGBM':
    best_model = lgb.LGBMRegressor(**lgb_params_final)
    best_model.fit(X, y)
elif best_single == 'XGBoost':
    best_model = xgb.XGBRegressor(**xgb_params_final)
    best_model.fit(X, y)
elif best_single == 'HistGBR':
    best_model = HistGradientBoostingRegressor(**hgb_params_final)
    best_model.fit(X, y)
else:
    best_model = ExtraTreesRegressor(**et_params_final)
    best_model.fit(X, y)

joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
print(f"Saved best model ({best_single}) to {OUTPUT_DIR}/best_model.pkl")

# Save all hyperparameters
all_params = {
    'CatBoost': cb_best_params,
    'LightGBM': lgb_best_params,
    'XGBoost': xgb_best_params,
    'HistGBR': hgb_best_params,
    'ExtraTrees': et_best_params,
}
joblib.dump(all_params, os.path.join(OUTPUT_DIR, 'all_best_params.pkl'))
print(f"Saved all best parameters to {OUTPUT_DIR}/all_best_params.pkl")

# ============================================================================
# 16. GENERATE SUBMISSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 16: GENERATE SUBMISSION")
print("=" * 80)

# Clip predictions to valid range (demand >= 0)
final_test_preds = np.clip(final_test_preds, 0, None)

submission = pd.DataFrame({
    'Index': test_index,
    'demand': final_test_preds
})

# Validate submission
print(f"Submission shape: {submission.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
assert submission.shape[0] == sample_sub.shape[0], f"Row mismatch! {submission.shape[0]} vs {sample_sub.shape[0]}"
assert list(submission.columns) == list(sample_sub.columns), f"Column mismatch! {list(submission.columns)} vs {list(sample_sub.columns)}"
print("✓ Submission shape matches sample_submission.csv")

print(f"\nSubmission statistics:")
print(submission['demand'].describe())

# Check for any issues
print(f"\nNaN in submission: {submission['demand'].isnull().sum()}")
print(f"Negative values: {(submission['demand'] < 0).sum()}")
print(f"Index range: [{submission['Index'].min()}, {submission['Index'].max()}]")

submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"\n✓ Saved submission to {OUTPUT_DIR}/submission.csv")

# Also save to root for easy access
submission.to_csv('submission.csv', index=False)
print(f"✓ Saved submission to ./submission.csv")

# ============================================================================
# 17. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
Competition: Traffic Demand Prediction
Metric: score = max(0, 100 * R²)

Best Solution: {best_solution_name}
Best CV Score: {final_score:.4f}
Best OOF R²: {r2_score(y, final_oof_preds):.6f}

Individual Model Scores (Optimized):
  CatBoost:   {cb_overall:.4f}
  LightGBM:   {lgb_overall:.4f}
  XGBoost:    {xgb_overall:.4f}
  HistGBR:    {hgb_overall:.4f}
  ExtraTrees: {et_overall:.4f}

Ensemble Scores:
  3-model weighted: {score_3:.4f}
  4-model weighted: {score_4:.4f}
  5-model weighted: {score_5:.4f}
  3-model average:  {avg_score_3:.4f}
  5-model average:  {avg_score_5:.4f}

Features Used: {len(feature_cols_final)}
Optuna Trials Per Model: {OPTUNA_TRIALS}
CV Strategy: {N_FOLDS}-Fold Stratified KFold

Output Files:
  - submission.csv
  - {OUTPUT_DIR}/submission.csv
  - {OUTPUT_DIR}/model_comparison.csv
  - {OUTPUT_DIR}/feature_importance.csv
  - {OUTPUT_DIR}/best_model.pkl
  - {OUTPUT_DIR}/all_best_params.pkl
  - {OUTPUT_DIR}/eda_overview.png
  - {OUTPUT_DIR}/correlation_matrix.png
  - {OUTPUT_DIR}/demand_by_category.png
  - {OUTPUT_DIR}/shap_summary.png
  - {OUTPUT_DIR}/shap_importance_bar.png
  - {OUTPUT_DIR}/feature_importance_comparison.png
""")

print("=" * 80)
print("SOLUTION COMPLETE!")
print("=" * 80)
