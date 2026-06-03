"""
Steps 9–11 — Train LightGBM, XGBoost, and CatBoost models.
Includes dual validation: temporal (Day48→Day49) + random 80/20 split.
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def _score(y_true, y_pred):
    """Competition scoring: max(0, 100 × R²)."""
    return max(0, 100 * r2_score(y_true, y_pred))


def split_data(train_df, X, y):
    """Create both temporal and random validation splits."""
    # Primary: Day 48 → Day 49 temporal split
    d48_mask = train_df["day"] == 48
    d49_mask = train_df["day"] == 49
    X_d48, y_d48 = X[d48_mask], y[d48_mask]
    X_d49, y_d49 = X[d49_mask], y[d49_mask]

    # Secondary: 80/20 random split (as prompt specifies)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Temporal split  — Train (Day48): {X_d48.shape} | Val (Day49): {X_d49.shape}")
    print(f"Random split    — Train: {X_tr.shape} | Val: {X_val.shape}")

    return {
        "temporal": (X_d48, y_d48, X_d49, y_d49),
        "random": (X_tr, y_tr, X_val, y_val),
    }


def train_lightgbm(splits):
    """Step 9 — Train LightGBM on the random split."""
    X_tr, y_tr, X_val, y_val = splits["random"]

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
    )

    # Scores on both splits
    rand_score = _score(y_val, model.predict(X_val))

    X_d48, y_d48, X_d49, y_d49 = splits["temporal"]
    temp_score = _score(y_d49, model.predict(X_d49))

    print(f"\n🟢 LightGBM Random Val Score  : {rand_score:.4f}")
    print(f"🟢 LightGBM Temporal Val Score: {temp_score:.4f}")

    return model, rand_score


def train_xgboost(splits):
    """Step 10 — Train XGBoost on the random split."""
    X_tr, y_tr, X_val, y_val = splits["random"]

    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=150,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=300,
    )

    rand_score = _score(y_val, model.predict(X_val))

    X_d48, y_d48, X_d49, y_d49 = splits["temporal"]
    temp_score = _score(y_d49, model.predict(X_d49))

    print(f"\n🟠 XGBoost Random Val Score  : {rand_score:.4f}")
    print(f"🟠 XGBoost Temporal Val Score: {temp_score:.4f}")

    return model, rand_score


def train_catboost(splits):
    """Step 11 — Train CatBoost on the random split."""
    X_tr, y_tr, X_val, y_val = splits["random"]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        random_strength=0.3,
        bagging_temperature=0.5,
        od_type="Iter",
        od_wait=150,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=300,
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    rand_score = _score(y_val, model.predict(X_val))

    X_d48, y_d48, X_d49, y_d49 = splits["temporal"]
    temp_score = _score(y_d49, model.predict(X_d49))

    print(f"\n🔵 CatBoost Random Val Score  : {rand_score:.4f}")
    print(f"🔵 CatBoost Temporal Val Score: {temp_score:.4f}")

    return model, rand_score


def train_all_models(train_df, X, y):
    """Execute Steps 9–11: train all three models."""
    splits = split_data(train_df, X, y)

    lgb_model, lgb_score = train_lightgbm(splits)
    print("✅ STEP 9 DONE\n")

    xgb_model, xgb_score = train_xgboost(splits)
    print("✅ STEP 10 DONE\n")

    cat_model, cat_score = train_catboost(splits)
    print("✅ STEP 11 DONE\n")

    models = {
        "lgb": (lgb_model, lgb_score),
        "xgb": (xgb_model, xgb_score),
        "cat": (cat_model, cat_score),
    }

    return models, splits
