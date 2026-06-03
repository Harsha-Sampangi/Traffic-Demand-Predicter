#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import joblib
import json
from r2_fe_utils import get_r2_data

print("=" * 80)
print("ROUND 2: STEP 2 & 3 - ADAPTIVE OPTUNA TUNING")
print("=" * 80)

# Load encoded data
X, y, X_test, feature_cols, _ = get_r2_data(native_cat=False)
train_raw = pd.read_csv('dataset/train.csv')
train_mask = train_raw['day'] == 48
val_mask = train_raw['day'] == 49

X_trn = X[train_mask]
y_trn_log = np.log1p(y[train_mask])
X_val = X[val_mask]
y_val_raw = y[val_mask]

# Load native categorical data for CatBoost Native
X_nat, y_nat, X_test_nat, feature_cols_nat, cat_features = get_r2_data(native_cat=True)
X_trn_nat = X_nat[train_mask]
X_val_nat = X_nat[val_mask]

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_trn, y_trn_log)
    preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
    return max(0, 100 * r2_score(y_val_raw, preds))

def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_trn, y_trn_log)
    preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
    return max(0, 100 * r2_score(y_val_raw, preds))

def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }
    model = CatBoostRegressor(**params)
    model.fit(X_trn, y_trn_log)
    preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
    return max(0, 100 * r2_score(y_val_raw, preds))

def cat_nat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,
        'cat_features': cat_features
    }
    model = CatBoostRegressor(**params)
    model.fit(X_trn_nat, y_trn_log)
    preds = np.clip(np.expm1(model.predict(X_val_nat)), 0, 1)
    return max(0, 100 * r2_score(y_val_raw, preds))

studies = {
    'xgb': optuna.create_study(direction='maximize'),
    'lgb': optuna.create_study(direction='maximize'),
    'cat': optuna.create_study(direction='maximize'),
    'cat_nat': optuna.create_study(direction='maximize')
}
objectives = {
    'xgb': xgb_objective,
    'lgb': lgb_objective,
    'cat': cat_objective,
    'cat_nat': cat_nat_objective
}

print("Running Phase 1: 30 trials per model (120 total)...")
for name in studies.keys():
    print(f"  Tuning {name}...")
    studies[name].optimize(objectives[name], n_trials=30)

scores = {name: study.best_value for name, study in studies.items()}
print(f"\nPhase 1 Results: {scores}")

# Allocate remaining 180 trials adaptively based on softmax of scores
total_score = sum(np.exp(s - max(scores.values())) for s in scores.values()) # stable softmax
allocations = {name: int(180 * (np.exp(score - max(scores.values())) / total_score)) for name, score in scores.items()}
print(f"Phase 2 Allocations (180 remaining): {allocations}")

for name in studies.keys():
    if allocations[name] > 0:
        print(f"  Continuing {name} with {allocations[name]} trials...")
        studies[name].optimize(objectives[name], n_trials=allocations[name])

print("\n--- FINAL OPTUNA RESULTS ---")
best_overall = None
best_overall_score = -1
for name, study in studies.items():
    print(f"{name} Best Score: {study.best_value:.4f}")
    if study.best_value > best_overall_score:
        best_overall_score = study.best_value
        best_overall = name
        
# Step 3 Check: Encoded vs Native CatBoost
print(f"\nCatBoost (Encoded) vs CatBoost (Native):")
if studies['cat_nat'].best_value > studies['cat'].best_value:
    print("  -> Native CatBoost is better! Keeping Native.")
    cat_to_use = 'cat_nat'
else:
    print("  -> Encoded CatBoost is better! Keeping Encoded.")
    cat_to_use = 'cat'

# Save OOF Predictions and Models
print("\nGenerating final OOF and Test predictions for XGB, LGB, and best CAT...")

models_to_train = ['xgb', 'lgb', cat_to_use]
y_log_full = np.log1p(y)
sub_idx = pd.read_csv('dataset/test.csv')[['Index']]
oof_idx = train_raw.loc[val_mask, ['Index']]

for m_name in models_to_train:
    print(f"  Training {m_name}...")
    params = studies[m_name].best_params.copy()
    
    if m_name == 'xgb':
        params.update({'random_state': 42, 'n_jobs': -1, 'tree_method': 'hist'})
        model = xgb.XGBRegressor(**params)
        model.fit(X_trn, y_trn_log)
        oof_preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
        model_full = xgb.XGBRegressor(**params)
        model_full.fit(X, y_log_full)
        test_preds = np.clip(np.expm1(model_full.predict(X_test)), 0, 1)
        joblib.dump(model_full, 'best_xgb.pkl')
        
    elif m_name == 'lgb':
        params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'bagging_freq': 1})
        model = lgb.LGBMRegressor(**params)
        model.fit(X_trn, y_trn_log)
        oof_preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
        model_full = lgb.LGBMRegressor(**params)
        model_full.fit(X, y_log_full)
        test_preds = np.clip(np.expm1(model_full.predict(X_test)), 0, 1)
        joblib.dump(model_full, 'best_lgbm.pkl')
        
    else: # cat or cat_nat
        params.update({'random_seed': 42, 'verbose': False, 'thread_count': -1})
        if m_name == 'cat_nat':
            params['cat_features'] = cat_features
            model = CatBoostRegressor(**params)
            model.fit(X_trn_nat, y_trn_log)
            oof_preds = np.clip(np.expm1(model.predict(X_val_nat)), 0, 1)
            model_full = CatBoostRegressor(**params)
            model_full.fit(X_nat, y_log_full)
            test_preds = np.clip(np.expm1(model_full.predict(X_test_nat)), 0, 1)
            joblib.dump(model_full, 'best_native_cat.pkl')
        else:
            model = CatBoostRegressor(**params)
            model.fit(X_trn, y_trn_log)
            oof_preds = np.clip(np.expm1(model.predict(X_val)), 0, 1)
            model_full = CatBoostRegressor(**params)
            model_full.fit(X, y_log_full)
            test_preds = np.clip(np.expm1(model_full.predict(X_test)), 0, 1)
            joblib.dump(model_full, 'best_cat.pkl')
            
    oof_df = oof_idx.copy()
    oof_df['demand'] = oof_preds
    oof_df.to_csv(f'oof_{m_name[:3]}.csv', index=False)
    
    test_df = sub_idx.copy()
    test_df['demand'] = test_preds
    test_df.to_csv(f'test_{m_name[:3]}.csv', index=False)

# Save dict of which cat we used so the next script knows
with open('cat_model_type.json', 'w') as f:
    json.dump({'cat_type': cat_to_use}, f)

print("DONE.")
