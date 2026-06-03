#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 2: TARGET TRANSFORMATION EXPERIMENT")
print("=" * 80)

# Load data
X, y, X_test, feature_cols = get_processed_data()
train_raw = pd.read_csv('dataset/train.csv')

# Use strictly the Time-Series split
train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask]
y_trn_raw = y[train_mask]
y_trn_log = np.log1p(y_trn_raw)

X_val = X[val_mask]
y_val = y[val_mask] # Ground truth is always raw

print(f"Train size: {len(X_trn)}, Val size: {len(X_val)}")

results = []

def run_experiment(model_name, model_fn):
    print(f"\n--- {model_name} ---")
    
    # Experiment A: Raw Target
    model_raw = model_fn()
    model_raw.fit(X_trn, y_trn_raw)
    preds_raw = model_raw.predict(X_val)
    # Clip to valid range [0, 1]
    preds_raw = np.clip(preds_raw, 0, 1)
    r2_raw = r2_score(y_val, preds_raw)
    score_raw = max(0, 100 * r2_raw)
    print(f"  Exp A (Raw Target): R²={r2_raw:.6f} | Score={score_raw:.2f}")
    
    # Experiment B: Log1p Target
    model_log = model_fn()
    model_log.fit(X_trn, y_trn_log)
    preds_log_space = model_log.predict(X_val)
    preds_exp = np.expm1(preds_log_space)
    # Clip to valid range [0, 1]
    preds_exp = np.clip(preds_exp, 0, 1)
    r2_log = r2_score(y_val, preds_exp)
    score_log = max(0, 100 * r2_log)
    print(f"  Exp B (Log Target): R²={r2_log:.6f} | Score={score_log:.2f}")
    
    diff = score_log - score_raw
    print(f"  Difference (Log - Raw): {diff:+.2f} points")
    
    results.append({
        'Model': model_name,
        'Raw_Score': score_raw,
        'Log_Score': score_log,
        'Diff': diff
    })

# Run for XGBoost
run_experiment(
    "XGBoost", 
    lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
)

# Run for LightGBM
run_experiment(
    "LightGBM", 
    lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
)

print("\n" + "=" * 80)
print("TARGET TRANSFORMATION REPORT")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

avg_diff = results_df['Diff'].mean()
if avg_diff > 0.5:
    print("\nRECOMMENDATION: Use log1p transformation. It stabilizes variance and provides a clear score boost.")
elif avg_diff < -0.5:
    print("\nRECOMMENDATION: Use raw target. Log transformation significantly degrades performance on the evaluation metric.")
else:
    print("\nRECOMMENDATION: Both methods perform similarly. Recommend using raw target for simplicity or ensembling both.")
    
print("DONE.")
