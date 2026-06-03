#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import optuna
from final_fe_utils import get_final_data

print("=" * 80)
print("STEP 4B: ADVANCED CATBOOST OPTIMIZATION")
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
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_trn, y_trn_log)
    
    preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
    score = max(0, 100 * r2_score(y_val_raw, preds))
    return score

study = optuna.create_study(direction='maximize', study_name="cat_tuning")
print("\nStarting Optuna trials for CatBoost (15 trials for speed)...")
study.optimize(objective, n_trials=15)

print("\n--- CatBoost Best Params ---")
print(study.best_params)
print(f"Best Val Score: {study.best_value:.4f}")

# Train best model for saving or ensembling later
print("\nTraining final CatBoost model with best params on full data...")
best_model = CatBoostRegressor(**study.best_params, random_seed=42, verbose=False, thread_count=-1)
y_log = np.log1p(y)
best_model.fit(X, y_log)

# Generate Test Predictions
preds_test = np.clip(np.expm1(best_model.predict(X_test)), 0, 1)
sub = pd.read_csv('dataset/test.csv')[['Index']]
sub['demand'] = preds_test
sub.to_csv('cat_best_submission.csv', index=False)
print("Saved cat_best_submission.csv")

print("DONE.")
