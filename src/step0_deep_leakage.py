#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

print("=" * 80)
print("STEP 0: DEEP LEAKAGE INVESTIGATION")
print("=" * 80)

DATA_DIR = 'dataset'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Create stratify bins (as done in previous script)
TARGET = 'demand'
y = train[TARGET].values
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(train, y_bins))

fold_idx = 0
train_idx, val_idx = folds[fold_idx]

train_fold = train.iloc[train_idx]
val_fold = train.iloc[val_idx]

print("\n--- Analysing Fold 1 vs Fold 1 Val ---")
print(f"Train size: {len(train_fold)}, Val size: {len(val_fold)}")

# 1. Geohash overlap
train_geohashes = set(train_fold['geohash'].unique())
val_geohashes = set(val_fold['geohash'].unique())
geohash_overlap = val_geohashes.intersection(train_geohashes)
print(f"\nGeohash overlap:")
print(f"  Val geohashes: {len(val_geohashes)}")
print(f"  Overlap with Train: {len(geohash_overlap)}")
print(f"  Overlap %: {len(geohash_overlap) / len(val_geohashes) * 100:.2f}%")

# 2. Timestamp overlap
train_timestamps = set(train_fold['timestamp'].unique())
val_timestamps = set(val_fold['timestamp'].unique())
ts_overlap = val_timestamps.intersection(train_timestamps)
print(f"\nTimestamp overlap:")
print(f"  Val timestamps: {len(val_timestamps)}")
print(f"  Overlap with Train: {len(ts_overlap)}")
print(f"  Overlap %: {len(ts_overlap) / len(val_timestamps) * 100:.2f}%")

# 3. Geohash + Timestamp combinations
train_fold['geo_ts'] = train_fold['geohash'] + "_" + train_fold['timestamp']
val_fold['geo_ts'] = val_fold['geohash'] + "_" + val_fold['timestamp']
test['geo_ts'] = test['geohash'] + "_" + test['timestamp']

train_geo_ts = set(train_fold['geo_ts'].unique())
val_geo_ts = set(val_fold['geo_ts'].unique())
geo_ts_overlap = val_geo_ts.intersection(train_geo_ts)
print(f"\nGeohash + Timestamp overlap:")
print(f"  Val combinations: {len(val_geo_ts)}")
print(f"  Overlap with Train: {len(geo_ts_overlap)}")
print(f"  Overlap %: {len(geo_ts_overlap) / len(val_geo_ts) * 100:.2f}%")

# Now check train vs test combinations! This is critical for leakage.
train_all_geo_ts = set(train['geohash'] + "_" + train['timestamp'])
test_geo_ts = set(test['geo_ts'].unique())
train_test_overlap = test_geo_ts.intersection(train_all_geo_ts)
print(f"\nGeohash + Timestamp overlap (Train vs Test):")
print(f"  Test combinations: {len(test_geo_ts)}")
print(f"  Overlap with Full Train: {len(train_test_overlap)}")
print(f"  Overlap %: {len(train_test_overlap) / len(test_geo_ts) * 100:.2f}%")

# 4. Distributions
print("\n--- Distribution Comparison (Train Fold vs Val Fold) ---")
print("RoadType Train Fold:")
print(train_fold['RoadType'].value_counts(normalize=True).round(4))
print("RoadType Val Fold:")
print(val_fold['RoadType'].value_counts(normalize=True).round(4))

print("\nWeather Train Fold:")
print(train_fold['Weather'].value_counts(normalize=True).round(4))
print("Weather Val Fold:")
print(val_fold['Weather'].value_counts(normalize=True).round(4))

# 5. Potential Duplicate Patterns
# Let's see if demand is the exact same for the exact same geohash + timestamp across different days?
# The dataset only has Day 48 and Day 49. Are there geo_ts that exist in both?
train['geo_ts'] = train['geohash'] + "_" + train['timestamp']
duplicates = train[train.duplicated(subset=['geo_ts'], keep=False)].sort_values('geo_ts')
print(f"\n--- Potential Duplicate Patterns ---")
print(f"Number of rows with duplicate (geohash, timestamp) in full train: {len(duplicates)}")
if len(duplicates) > 0:
    print(f"Example duplicates:")
    print(duplicates[['day', 'geohash', 'timestamp', 'demand']].head(6))
else:
    print("No rows share the exact same geohash + timestamp in the training data.")

print("\n" + "=" * 80)
print("LEAKAGE RISK REPORT")
print("=" * 80)
if len(train_test_overlap) / len(test_geo_ts) < 0.1:
    print("WARNING: Low overlap between train and test geohash+timestamp combinations.")
    print("If validation has high overlap (e.g., 0%), but train vs test has 0%, random k-fold is leaking.")
    print("Wait, if validation has high overlap, it's leaking. If overlap is low, it's safer.")

print("\nDONE.")
