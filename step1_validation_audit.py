#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import xgboost as xgb
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 1: VALIDATION AUDIT")
print("=" * 80)

# Load data
X, y, X_test, feature_cols = get_processed_data()
train_raw = pd.read_csv('dataset/train.csv')

print(f"Loaded {len(X)} rows for training.")
print("Comparing 5-Fold Stratified CV against Time-Series (Day-based) Split.")

# 1. 5-Fold Stratified CV (current method)
print("\n--- Current Strategy: 5-Fold Stratified CV ---")
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y_bins)):
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1
    )
    model.fit(X_trn, y_trn)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    cv_scores.append(r2)
    print(f"  Fold {fold+1} R²: {r2:.4f}")

mean_cv_r2 = np.mean(cv_scores)
print(f"  Mean 5-Fold CV R²: {mean_cv_r2:.4f} (Score: {max(0, 100*mean_cv_r2):.2f})")

# 2. Strict Time-Series Validation (Train: Day 48, Val: Day 49)
print("\n--- Proposed Strategy: Time-Series Split (Day 48 -> Day 49) ---")
train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn_ts = X[train_mask]
y_trn_ts = y[train_mask]
X_val_ts = X[val_mask]
y_val_ts = y[val_mask]

print(f"  Train (Day 48): {len(X_trn_ts)} rows")
print(f"  Val (Day 49):   {len(X_val_ts)} rows")

model_ts = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=6,
    random_state=42, n_jobs=-1
)
model_ts.fit(X_trn_ts, y_trn_ts)
preds_ts = model_ts.predict(X_val_ts)
ts_r2 = r2_score(y_val_ts, preds_ts)
print(f"  Time-Series Val R²: {ts_r2:.4f} (Score: {max(0, 100*ts_r2):.2f})")

print("\n" + "=" * 80)
print("VALIDATION AUDIT REPORT")
print("=" * 80)
diff = mean_cv_r2 - ts_r2
print(f"5-Fold CV Score:      {max(0, 100*mean_cv_r2):.2f}")
print(f"Time-Series CV Score: {max(0, 100*ts_r2):.2f}")
print(f"Difference:           {diff*100:.2f} points")

if diff > 0.02:
    print("\nCONCLUSION: High risk of temporal leakage detected.")
    print("Random K-Fold interpolates between timestamps, inflating the score.")
    print("Since the Test set contains unseen Day 49 timestamps, Time-Series Split is significantly more trustworthy.")
else:
    print("\nCONCLUSION: Random K-Fold is stable.")

print("DONE.")
