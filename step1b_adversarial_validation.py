#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from fe_utils import get_processed_data

print("=" * 80)
print("STEP 1B: ADVERSARIAL VALIDATION")
print("=" * 80)

# Load processed data
X_train, y, X_test, feature_cols = get_processed_data()

# Add is_test label
X_train['is_test'] = 0
X_test['is_test'] = 1

# Combine datasets
X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_adv = X_combined['is_test'].values
X_adv = X_combined.drop('is_test', axis=1)

print(f"Combined data shape: {X_adv.shape}")
print(f"Target distribution (is_test):\n{pd.Series(y_adv).value_counts(normalize=True)}")

# Stratified KFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros(len(y_adv))
oof_preds_lgb = np.zeros(len(y_adv))

feature_importances_xgb = np.zeros(X_adv.shape[1])
feature_importances_lgb = np.zeros(X_adv.shape[1])

print("\n--- Training Adversarial XGBoost ---")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
    X_trn, X_val = X_adv.iloc[trn_idx], X_adv.iloc[val_idx]
    y_trn, y_val = y_adv[trn_idx], y_adv[val_idx]
    
    clf = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, n_jobs=-1, eval_metric='auc'
    )
    clf.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds_xgb[val_idx] = clf.predict_proba(X_val)[:, 1]
    feature_importances_xgb += clf.feature_importances_ / 5
    auc = roc_auc_score(y_val, oof_preds_xgb[val_idx])
    print(f"Fold {fold+1} AUC: {auc:.4f}")

auc_xgb = roc_auc_score(y_adv, oof_preds_xgb)
print(f"XGBoost Overall Adversarial AUC: {auc_xgb:.4f}")

print("\n--- Training Adversarial LightGBM ---")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
    X_trn, X_val = X_adv.iloc[trn_idx], X_adv.iloc[val_idx]
    y_trn, y_val = y_adv[trn_idx], y_adv[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    clf.fit(X_trn, y_trn)
    
    oof_preds_lgb[val_idx] = clf.predict_proba(X_val)[:, 1]
    feature_importances_lgb += clf.feature_importances_ / 5
    auc = roc_auc_score(y_val, oof_preds_lgb[val_idx])
    print(f"Fold {fold+1} AUC: {auc:.4f}")

auc_lgb = roc_auc_score(y_adv, oof_preds_lgb)
print(f"LightGBM Overall Adversarial AUC: {auc_lgb:.4f}")

if auc_xgb > 0.70 or auc_lgb > 0.70:
    print("\nWARNING: Significant train/test distribution shift detected (AUC > 0.70)!")
    
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance_XGB': feature_importances_xgb,
        'Importance_LGB': feature_importances_lgb
    })
    fi_df['Avg_Importance'] = (fi_df['Importance_XGB'] + fi_df['Importance_LGB']) / 2
    fi_df = fi_df.sort_values('Avg_Importance', ascending=False)
    
    print("\nTop 15 shifted features:")
    print(fi_df.head(15))
    
    print("\nRecommendations for handling shift:")
    print("- Drop top shifted features if they do not contribute significantly to demand prediction.")
    print("- Ensure cross-validation strategy mimics the test set distribution (e.g. stratify by shifted features or use group K-fold).")
    print("- Do NOT use Pseudo Labeling if the shift is severe on critical features.")
else:
    print("\nNo significant distribution shift detected (AUC <= 0.70).")
    print("Train and Test distributions appear similar.")
    print("Pseudo labeling is safe to use in later steps.")

print("\nDONE.")
