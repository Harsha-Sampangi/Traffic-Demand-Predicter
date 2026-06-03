#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
import optuna
from final_fe_utils import get_final_data

print("=" * 80)
print("STEP 4: ADVANCED LIGHTGBM OPTIMIZATION")
print("=" * 80)

# Load final optimized data
X, y, X_test, feature_cols = get_final_data()
train_raw = pd.read_csv('dataset/train.csv')

train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask]
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask]
y_val_raw = y[val_mask]

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_trn, y_trn_log)
    
    preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
    score = max(0, 100 * r2_score(y_val_raw, preds))
    return score

study = optuna.create_study(direction='maximize', study_name="lgb_tuning")
print("\nStarting Optuna trials for LightGBM (15 trials for speed)...")
study.optimize(objective, n_trials=15)

print("\n--- LightGBM Best Params ---")
print(study.best_params)
print(f"Best Val Score: {study.best_value:.4f}")

# Train best model for saving or ensembling later
print("\nTraining final LightGBM model with best params on full data...")
best_model = lgb.LGBMRegressor(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
y_log = np.log1p(y)
best_model.fit(X, y_log)

# Generate Test Predictions
preds_test = np.clip(np.expm1(best_model.predict(X_test)), 0, 1)
sub = pd.read_csv('dataset/test.csv')[['Index']]
sub['demand'] = preds_test
sub.to_csv('lgb_best_submission.csv', index=False)
print("Saved lgb_best_submission.csv")

print("DONE.")
