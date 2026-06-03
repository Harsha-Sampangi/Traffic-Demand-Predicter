#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from final_fe_utils import get_final_data

print("=" * 80)
print("STEP 8: FINAL MODEL ENSEMBLE")
print("=" * 80)

# 1. Retrain XGBoost since it crashed during the submission generation step
print("Retraining Final XGBoost Model with best params...")
X, y, X_test, feature_cols = get_final_data()
y_log = np.log1p(y)

xgb_best_params = {
    'n_estimators': 313,
    'learning_rate': 0.014810504066812792,
    'max_depth': 9,
    'min_child_weight': 7,
    'subsample': 0.501290975198775,
    'colsample_bytree': 0.9179927617220671,
    'reg_alpha': 3.0575834541430506e-05,
    'reg_lambda': 4.612722519966238e-06,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

model_xgb = xgb.XGBRegressor(**xgb_best_params)
model_xgb.fit(X, y_log)

preds_xgb = np.clip(np.expm1(model_xgb.predict(X_test)), 0, 1)

# Save XGBoost predictions
sub_xgb = pd.read_csv('dataset/test.csv')[['Index']]
sub_xgb['demand'] = preds_xgb
sub_xgb.to_csv('xgb_best_submission.csv', index=False)
print("Saved xgb_best_submission.csv")

# 2. Ensemble Predictions
print("\nEnsembling LightGBM, CatBoost, and XGBoost predictions...")
try:
    lgb_sub = pd.read_csv('lgb_best_submission.csv')
    cat_sub = pd.read_csv('cat_best_submission.csv')
    xgb_sub = pd.read_csv('xgb_best_submission.csv')
    
    # Simple average ensemble
    final_preds = (lgb_sub['demand'] + cat_sub['demand'] + xgb_sub['demand']) / 3.0
    
    final_sub = lgb_sub[['Index']].copy()
    final_sub['demand'] = final_preds
    final_sub.to_csv('final_submission.csv', index=False)
    
    print("\nSuccessfully generated final_submission.csv using an ensemble of 3 optimized models!")
except Exception as e:
    print(f"Error reading sub files for ensemble: {e}")

print("DONE.")
