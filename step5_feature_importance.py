#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import shap
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 5: FEATURE IMPORTANCE INVESTIGATION")
print("=" * 80)

# Load data
X, y, X_test, feature_cols = get_processed_data()
train_raw = pd.read_csv('dataset/train.csv')

# Use Time-Series split & Log1p target
train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask]
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask]
y_val_raw = y[val_mask]

print(f"Train size: {len(X_trn)}, Val size: {len(X_val)}")
print(f"Total features: {len(feature_cols)}")

# Train model
print("\nTraining LightGBM model for feature importance...")
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_trn, y_trn_log)

# Baseline score
preds_val = np.clip(np.expm1(model.predict(X_val)), 0, 1)
baseline_r2 = max(0, 100 * r2_score(y_val_raw, preds_val))
print(f"Baseline Score: {baseline_r2:.4f}")

# 1. Gain Importance
print("\nExtracting Gain Importance...")
gain_imp = model.booster_.feature_importance(importance_type='gain')
df_gain = pd.DataFrame({'Feature': feature_cols, 'Gain': gain_imp})
df_gain['Gain_Rank'] = df_gain['Gain'].rank(ascending=False)

# 2. Permutation Importance
print("Extracting Permutation Importance...")
def score_func(estimator, X_eval, y_eval):
    preds = np.clip(np.expm1(estimator.predict(X_eval)), 0, 1)
    return max(0, 100 * r2_score(y_eval, preds))

# Use small subset of val for speed if needed, but 7.8k rows is fast enough
perm_results = permutation_importance(model, X_val, y_val_raw, scoring=score_func, n_repeats=3, random_state=42, n_jobs=-1)
df_perm = pd.DataFrame({'Feature': feature_cols, 'Permutation': perm_results.importances_mean})
df_perm['Permutation_Rank'] = df_perm['Permutation'].rank(ascending=False)

# 3. SHAP Importance
print("Extracting SHAP Importance (subsampled for speed)...")
# Subsample for SHAP to avoid hanging
X_shap = X_val.sample(n=min(2000, len(X_val)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
shap_imp = np.abs(shap_values).mean(axis=0)
df_shap = pd.DataFrame({'Feature': feature_cols, 'SHAP': shap_imp})
df_shap['SHAP_Rank'] = df_shap['SHAP'].rank(ascending=False)

# Combine and Aggregate
df_imp = df_gain.merge(df_perm, on='Feature').merge(df_shap, on='Feature')
df_imp['Avg_Rank'] = (df_imp['Gain_Rank'] + df_imp['Permutation_Rank'] + df_imp['SHAP_Rank']) / 3
df_imp = df_imp.sort_values('Avg_Rank')

print("\n--- TOP 20 FEATURES ---")
print(df_imp.head(20)[['Feature', 'Avg_Rank', 'Gain', 'Permutation', 'SHAP']])

print("\n--- BOTTOM 20 FEATURES ---")
bottom_20 = df_imp.tail(20)
print(bottom_20[['Feature', 'Avg_Rank', 'Gain', 'Permutation', 'SHAP']])

# Feature Reduction Experiment
print("\n" + "=" * 80)
print("FEATURE REDUCTION EXPERIMENT")
print("=" * 80)

# Identify features that have <= 0 permutation importance (meaning they add noise or do nothing)
weak_features_perm = df_imp[df_imp['Permutation'] <= 0.001]['Feature'].tolist()
print(f"Found {len(weak_features_perm)} features with <=0.001 permutation importance.")
print("Weak features:", weak_features_perm)

if len(weak_features_perm) > 0:
    print(f"\nRetraining without {len(weak_features_perm)} weak features...")
    features_reduced = [c for c in feature_cols if c not in weak_features_perm]
    X_trn_red = X_trn[features_reduced]
    X_val_red = X_val[features_reduced]
    
    model_red = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    model_red.fit(X_trn_red, y_trn_log)
    preds_val_red = np.clip(np.expm1(model_red.predict(X_val_red)), 0, 1)
    reduced_r2 = max(0, 100 * r2_score(y_val_raw, preds_val_red))
    
    print(f"Baseline Score: {baseline_r2:.4f}")
    print(f"Reduced  Score: {reduced_r2:.4f}")
    diff = reduced_r2 - baseline_r2
    print(f"Difference: {diff:+.4f} points")
    
    if diff > 0.05:
        print("\nRECOMMENDATION: Remove these weak features for subsequent models.")
    else:
        print("\nRECOMMENDATION: Keep all features, dropping them did not significantly improve score.")
else:
    print("\nRECOMMENDATION: No weak features found to drop.")

print("DONE.")
