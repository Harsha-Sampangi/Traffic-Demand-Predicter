#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

print("=" * 80)
print("ROUND 2: ADVANCED ENSEMBLING & ANALYSIS")
print("=" * 80)

# 1. Load Data
train_raw = pd.read_csv('dataset/train.csv')
val_mask = train_raw['day'] == 49
y_val = train_raw.loc[val_mask, 'demand'].values

oof_xgb = pd.read_csv('oof_xgb.csv')['demand'].values
oof_lgb = pd.read_csv('oof_lgb.csv')['demand'].values
oof_cat = pd.read_csv('oof_cat.csv')['demand'].values

test_xgb = pd.read_csv('test_xgb.csv')['demand'].values
test_lgb = pd.read_csv('test_lgb.csv')['demand'].values
test_cat = pd.read_csv('test_cat.csv')['demand'].values

df_oof = pd.DataFrame({'XGB': oof_xgb, 'LGB': oof_lgb, 'CAT': oof_cat})

# --- STEP 4: PREDICTION CORRELATION ---
print("\n--- STEP 4: Prediction Correlation Analysis ---")
pearson_corr = df_oof.corr(method='pearson')
spearman_corr = df_oof.corr(method='spearman')

print("Pearson Correlation:")
print(pearson_corr)
print("\nSpearman Correlation:")
print(spearman_corr)

corr_report = pd.DataFrame({
    'XGB_LGB_Pearson': [pearson_corr.loc['XGB', 'LGB']],
    'XGB_CAT_Pearson': [pearson_corr.loc['XGB', 'CAT']],
    'LGB_CAT_Pearson': [pearson_corr.loc['LGB', 'CAT']],
})
corr_report.to_csv('model_correlation_report.csv', index=False)

# --- STEP 5: ENSEMBLE WEIGHT OPTIMIZATION ---
print("\n--- STEP 5: Ensemble Weight Optimization ---")
def r2_objective(weights):
    w_xgb, w_lgb, w_cat = weights
    blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
    return -r2_score(y_val, blend)

# Constraints: weights sum to 1, non-negative
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1), (0, 1), (0, 1)]
init_w = [1/3, 1/3, 1/3]

# Equal weight baseline
equal_r2 = max(0, 100 * r2_score(y_val, (oof_xgb + oof_lgb + oof_cat) / 3.0))

res = minimize(r2_objective, init_w, bounds=bounds, constraints=cons)
opt_w = res.x
opt_blend = opt_w[0] * oof_xgb + opt_w[1] * oof_lgb + opt_w[2] * oof_cat
opt_r2 = max(0, 100 * r2_score(y_val, opt_blend))

print(f"Equal Weights R²:     {equal_r2:.4f}")
print(f"Optimized Weights R²: {opt_r2:.4f}")
print(f"Optimal Weights: XGB={opt_w[0]:.4f}, LGB={opt_w[1]:.4f}, CAT={opt_w[2]:.4f}")

with open('ensemble_weights.json', 'w') as f:
    json.dump({'w_xgb': opt_w[0], 'w_lgb': opt_w[1], 'w_cat': opt_w[2]}, f)

# --- STEP 6: RANK-BASED ENSEMBLE ---
print("\n--- STEP 6: Rank-Based Ensemble ---")
rank_xgb = rankdata(oof_xgb) / len(oof_xgb)
rank_lgb = rankdata(oof_lgb) / len(oof_lgb)
rank_cat = rankdata(oof_cat) / len(oof_cat)

equal_rank = (rank_xgb + rank_lgb + rank_cat) / 3.0
# Map back to demand distribution (roughly) using median of sorted values
sorted_y = np.sort(y_val)
rank_preds = np.interp(equal_rank, np.linspace(0, 1, len(sorted_y)), sorted_y)
rank_r2 = max(0, 100 * r2_score(y_val, rank_preds))
print(f"Rank-Based R²: {rank_r2:.4f}")

# --- STEP 7: STACKING WITH ROLLING SPLITS ---
print("\n--- STEP 7: Stacking (Rolling Day 48 Pseudo-OOFs) ---")
# To do this safely without leaking, we re-train the models on rolling slices of Day 48.
from r2_fe_utils import get_r2_data

# Read which CatBoost model was used
with open('cat_model_type.json', 'r') as f:
    cat_type = json.load(f)['cat_type']

X, y, _, _, _ = get_r2_data(native_cat=False)
train_mask = train_raw['day'] == 48
X_trn = X[train_mask].reset_index(drop=True)
y_trn_log = np.log1p(y[train_mask])

if cat_type == 'cat_nat':
    X_nat, _, _, _, cat_features = get_r2_data(native_cat=True)
    X_trn_cat = X_nat[train_mask].reset_index(drop=True)
else:
    X_trn_cat = X_trn.copy()

tscv = TimeSeriesSplit(n_splits=5)
meta_features = np.zeros((len(X_trn), 3))
meta_targets = np.zeros(len(X_trn))

# We need the model params. Let's load the best models and extract params
xgb_model = joblib.load('best_xgb.pkl')
lgb_model = joblib.load('best_lgbm.pkl')
cat_model = joblib.load('best_native_cat.pkl' if cat_type == 'cat_nat' else 'best_cat.pkl')

print("Generating temporal pseudo-OOFs...")
for split_idx, (trn_idx, val_idx) in enumerate(tscv.split(X_trn)):
    # XGB
    m_xgb = xgb.XGBRegressor(**xgb_model.get_params())
    m_xgb.fit(X_trn.iloc[trn_idx], y_trn_log[trn_idx])
    meta_features[val_idx, 0] = np.clip(np.expm1(m_xgb.predict(X_trn.iloc[val_idx])), 0, 1)
    
    # LGB
    m_lgb = lgb.LGBMRegressor(**lgb_model.get_params())
    m_lgb.fit(X_trn.iloc[trn_idx], y_trn_log[trn_idx])
    meta_features[val_idx, 1] = np.clip(np.expm1(m_lgb.predict(X_trn.iloc[val_idx])), 0, 1)
    
    # CAT
    m_cat = CatBoostRegressor(**cat_model.get_params())
    m_cat.fit(X_trn_cat.iloc[trn_idx], y_trn_log[trn_idx], verbose=False)
    meta_features[val_idx, 2] = np.clip(np.expm1(m_cat.predict(X_trn_cat.iloc[val_idx])), 0, 1)
    
    meta_targets[val_idx] = y_trn_log[val_idx]

# Drop the first fold since it's just used for training and has no OOF predictions
valid_mask = meta_features[:, 0] > 0
meta_X = meta_features[valid_mask]
meta_y = np.expm1(meta_targets[valid_mask])

# Day 49 Val features
meta_X_val = np.column_stack((oof_xgb, oof_lgb, oof_cat))

meta_models = {
    'Ridge': Ridge(alpha=1.0),
    'Linear': LinearRegression(),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

stack_results = {}
for name, meta_m in meta_models.items():
    meta_m.fit(meta_X, meta_y)
    stack_preds = np.clip(meta_m.predict(meta_X_val), 0, 1)
    r2 = max(0, 100 * r2_score(y_val, stack_preds))
    stack_results[name] = r2
    print(f"Stacking ({name}) R²: {r2:.4f}")

best_stack = max(stack_results, key=stack_results.get)
best_stack_score = stack_results[best_stack]
print(f"Best Meta-Model: {best_stack} ({best_stack_score:.4f})")

# --- STEP 10: FINAL MODEL SELECTION ---
print("\n--- STEP 10: Final Model Selection ---")

# Evaluate base models
base_xgb_r2 = max(0, 100 * r2_score(y_val, oof_xgb))
base_lgb_r2 = max(0, 100 * r2_score(y_val, oof_lgb))
base_cat_r2 = max(0, 100 * r2_score(y_val, oof_cat))

leaderboard = [
    {'Model': 'XGBoost', 'Validation Score': base_xgb_r2},
    {'Model': 'LightGBM', 'Validation Score': base_lgb_r2},
    {'Model': f'CatBoost ({cat_type})', 'Validation Score': base_cat_r2},
    {'Model': 'Equal Ensemble', 'Validation Score': equal_r2},
    {'Model': 'Optimized Ensemble', 'Validation Score': opt_r2},
    {'Model': 'Rank Ensemble', 'Validation Score': rank_r2},
    {'Model': f'Stacking ({best_stack})', 'Validation Score': best_stack_score}
]

lb_df = pd.DataFrame(leaderboard).sort_values('Validation Score', ascending=False)
print(lb_df)
lb_df.to_csv('leaderboard_comparison.csv', index=False)

# Generate Final Submission
best_approach = lb_df.iloc[0]['Model']
print(f"\nSelecting highest-scoring approach: {best_approach}")

if 'Optimized Ensemble' in best_approach:
    final_test_preds = opt_w[0] * test_xgb + opt_w[1] * test_lgb + opt_w[2] * test_cat
elif 'Equal Ensemble' in best_approach:
    final_test_preds = (test_xgb + test_lgb + test_cat) / 3.0
elif 'Stacking' in best_approach:
    meta_m = meta_models[best_stack]
    # retrain meta model on all day 48 + day 49
    # Wait, meta model trains on base models' OOF. To be robust, we just use it as is
    meta_X_test = np.column_stack((test_xgb, test_lgb, test_cat))
    final_test_preds = np.clip(meta_m.predict(meta_X_test), 0, 1)
elif 'Rank' in best_approach:
    # Rank ensemble on test
    rank_xgb_test = rankdata(test_xgb) / len(test_xgb)
    rank_lgb_test = rankdata(test_lgb) / len(test_lgb)
    rank_cat_test = rankdata(test_cat) / len(test_cat)
    equal_rank_test = (rank_xgb_test + rank_lgb_test + rank_cat_test) / 3.0
    final_test_preds = np.interp(equal_rank_test, np.linspace(0, 1, len(sorted_y)), sorted_y)
elif 'XGBoost' in best_approach:
    final_test_preds = test_xgb
elif 'LightGBM' in best_approach:
    final_test_preds = test_lgb
else:
    final_test_preds = test_cat

sub = pd.read_csv('dataset/test.csv')[['Index']]
sub['demand'] = final_test_preds
sub.to_csv('submission_final.csv', index=False)
print("Saved submission_final.csv")

print("DONE.")
