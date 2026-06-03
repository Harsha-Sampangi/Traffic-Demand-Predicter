"""
Steps 12–14, 16 — Weighted ensemble, test predictions, sanity checks,
submission generation, and full-data retrain.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


def _score(y_true, y_pred):
    return max(0, 100 * r2_score(y_true, y_pred))


def build_ensemble(models, splits, X_test, train_df, output_dir="output"):
    """Step 12 — Compute weighted ensemble from validation scores."""
    lgb_model, lgb_score = models["lgb"]
    xgb_model, xgb_score = models["xgb"]
    cat_model, cat_score = models["cat"]

    X_tr, y_tr, X_val, y_val = splits["random"]

    raw_scores = np.array([lgb_score, xgb_score, cat_score])
    raw_scores = np.maximum(raw_scores, 0)
    weights = raw_scores / raw_scores.sum()

    print(f"Weights → LGB: {weights[0]:.3f} | XGB: {weights[1]:.3f} | CAT: {weights[2]:.3f}")

    lgb_val = lgb_model.predict(X_val)
    xgb_val = xgb_model.predict(X_val)
    cat_val = cat_model.predict(X_val)

    ensemble_val = (
        weights[0] * lgb_val +
        weights[1] * xgb_val +
        weights[2] * cat_val
    )
    ensemble_score = _score(y_val, ensemble_val)

    print(f"\nScore summary:")
    print(f"  LightGBM  : {lgb_score:.4f}")
    print(f"  XGBoost   : {xgb_score:.4f}")
    print(f"  CatBoost  : {cat_score:.4f}")
    print(f"  Ensemble  : {ensemble_score:.4f}  ← final")

    return weights, ensemble_score


def generate_submission(models, weights, X_test, test_df, train_df, output_dir="output"):
    """Steps 13–14 — Generate and validate submission."""
    lgb_model, _ = models["lgb"]
    xgb_model, _ = models["xgb"]
    cat_model, _ = models["cat"]

    lgb_test = lgb_model.predict(X_test)
    xgb_test = xgb_model.predict(X_test)
    cat_test = cat_model.predict(X_test)

    final_preds = (
        weights[0] * lgb_test +
        weights[1] * xgb_test +
        weights[2] * cat_test
    )

    demand_min = train_df["demand"].min()
    demand_max = train_df["demand"].max()
    final_preds = np.clip(final_preds, demand_min, demand_max)

    print("Prediction range:", round(final_preds.min(), 6), "→", round(final_preds.max(), 6))
    print("NaN count:", np.isnan(final_preds).sum())

    submission = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": final_preds,
    })

    # Hard checks
    assert submission.shape == (41778, 2), f"Wrong shape: {submission.shape}"
    assert list(submission.columns) == ["Index", "demand"], "Wrong column names!"
    assert submission.isnull().sum().sum() == 0, "NaN values in submission!"
    assert (final_preds >= 0).all(), "Negative predictions found!"
    assert (final_preds <= 1).all(), "Predictions exceed 1.0!"

    out_path = f"{output_dir}/submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nsubmission.csv saved ✅")
    print("Shape          :", submission.shape)
    print("Column names   :", submission.columns.tolist())
    print("Demand range   :", round(submission["demand"].min(), 6), "→",
          round(submission["demand"].max(), 6))
    print("\nFirst 5 rows:")
    print(submission.head())

    return submission


def full_retrain_and_submit(models, weights, X, y, X_test, test_df, train_df,
                            output_dir="output"):
    """Step 16 — Retrain all models on full train data and generate v2 submission."""
    from src.feature_engineer import FEATURES

    lgb_model, _ = models["lgb"]
    xgb_model, _ = models["xgb"]
    cat_model, _ = models["cat"]

    # Retrain LightGBM
    best_lgb_iter = lgb_model.best_iteration_
    lgb_final = lgb.LGBMRegressor(
        n_estimators=best_lgb_iter,
        learning_rate=0.02, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42,
    )
    lgb_final.fit(X, y)

    # Retrain XGBoost
    best_xgb_iter = xgb_model.best_iteration
    xgb_final = xgb.XGBRegressor(
        n_estimators=best_xgb_iter,
        learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    xgb_final.fit(X, y)

    # Retrain CatBoost
    cat_final = CatBoostRegressor(
        iterations=cat_model.best_iteration_,
        learning_rate=0.02, depth=8,
        random_seed=42, verbose=0,
    )
    cat_final.fit(X, y)

    demand_min = train_df["demand"].min()
    demand_max = train_df["demand"].max()

    final_preds_v2 = np.clip(
        weights[0] * lgb_final.predict(X_test) +
        weights[1] * xgb_final.predict(X_test) +
        weights[2] * cat_final.predict(X_test),
        demand_min, demand_max,
    )

    submission_v2 = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": final_preds_v2,
    })
    out_path = f"{output_dir}/submission_v2_full_retrain.csv"
    submission_v2.to_csv(out_path, index=False)

    print(f"submission_v2_full_retrain.csv saved ✅")
    print("Use this file for your final upload — trained on 100% of available data.")

    return submission_v2
