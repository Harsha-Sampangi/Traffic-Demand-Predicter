# Traffic Demand Prediction — Antigravity IDE Prompt
### Paste this entire prompt into Antigravity. Follow every step in order.

---

## CONTEXT (read before running anything)

You are an expert ML engineer solving a traffic demand prediction competition.

**Dataset facts (already confirmed from EDA):**
- `train.csv` — 77,299 rows × 11 columns. Has the `demand` column.
- `test.csv`  — 41,778 rows × 10 columns. No `demand` — this is what you predict.
- `submission.csv` — must have exactly 2 columns: `Index` and `demand`, 41,778 rows.
- `demand` range: ~0.0000006 → 1.0 (right-skewed — 72% of rows are below 0.1)
- Timestamp format: `"H:MM"` string (e.g. `"2:15"`) — NOT a standard datetime.
- Train has days 48 and 49. Test has ONLY day 49, hours 2–13.
- Missing values: RoadType (600 train / 324 test), Temperature (2495 / 1349), Weather (797 / 431)
- 10 geohashes in test are completely unseen in train.
- RoadType is the single strongest predictor. Weather has near-zero signal.

**Scoring formula:**
```
score = max(0, 100 × R²_score(actual, predicted))
```
Perfect predictions = 100. Worse than guessing the mean = 0.

**Target: score = 100**
Strategy: 3-model stacked ensemble (LightGBM + XGBoost + CatBoost)
with target-encoded geohash and cyclical time features.

After completing each step, print: `✅ STEP N DONE` before proceeding.

---

## STEP 1 — Install all required libraries

```python
!pip install lightgbm xgboost catboost python-geohash scikit-learn pandas numpy --quiet
print("✅ STEP 1 DONE")
```

---

## STEP 2 — Load the data and confirm shapes

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub   = pd.read_csv("sample_submission.csv")

assert train.shape == (77299, 11), f"Unexpected train shape: {train.shape}"
assert test.shape  == (41778, 10), f"Unexpected test shape: {test.shape}"

print("Train shape :", train.shape)
print("Test shape  :", test.shape)
print("Sub columns :", sub.columns.tolist())
print("Train cols  :", train.columns.tolist())
print("\nMissing in train:\n", train.isnull().sum())
print("\nMissing in test:\n",  test.isnull().sum())
print("\nDemand stats:\n",     train["demand"].describe())
print("✅ STEP 2 DONE")
```

---

## STEP 3 — Parse timestamp (H:MM format — do NOT use pd.to_datetime)

```python
# Timestamp is a string like "2:15" or "14:30" — split manually
def parse_ts(ts):
    parts = str(ts).split(":")
    return int(parts[0]), int(parts[1])

for df in [train, test]:
    df[["hour", "minute"]] = df["timestamp"].apply(
        lambda x: pd.Series(parse_ts(x))
    )

# Verify
print("Train hours:", sorted(train["hour"].unique()))
print("Test hours: ", sorted(test["hour"].unique()))
# Expected: train = 0-23, test = 2-13 only
print("✅ STEP 3 DONE")
```

---

## STEP 4 — Fill missing values

```python
# Strategy:
# RoadType  → fill with mode (most frequent value = "Residential")
# Weather   → fill with mode (most frequent = "Sunny")
# Temperature → fill with median per geohash, then global median fallback

for df in [train, test]:
    df["RoadType"]    = df["RoadType"].fillna(df["RoadType"].mode()[0])
    df["Weather"]     = df["Weather"].fillna(df["Weather"].mode()[0])

# Temperature: smarter fill using location median
temp_median_by_geo = train.groupby("geohash")["Temperature"].median()

for df in [train, test]:
    df["Temperature"] = df["Temperature"].fillna(
        df["geohash"].map(temp_median_by_geo)
    )
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

print("Missing after fill — train:", train.isnull().sum().sum())
print("Missing after fill — test: ", test.isnull().sum().sum())
print("✅ STEP 4 DONE")
```

---

## STEP 5 — Feature engineering (highest impact step)

```python
import geohash as gh

# --- 5a. Decode geohash to lat / lon ---
def decode_geo(h):
    try:
        lat, lon = gh.decode(str(h))
        return float(lat), float(lon)
    except:
        return 0.0, 0.0

for df in [train, test]:
    coords = df["geohash"].apply(lambda x: pd.Series(decode_geo(x)))
    df["lat"] = coords[0]
    df["lon"]  = coords[1]

    # Geohash precision levels (zoom hierarchy)
    df["geo_p3"] = df["geohash"].str[:3]   # city-level zone
    df["geo_p4"] = df["geohash"].str[:4]   # district-level ~40km
    df["geo_p5"] = df["geohash"].str[:5]   # neighbourhood ~5km

    # --- 5b. Cyclical time encoding ---
    # Critical: sin/cos so the model knows 23h and 0h are neighbours
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]    / 24)
    df["minute_sin"]  = np.sin(2 * np.pi * df["minute"]  / 60)
    df["minute_cos"]  = np.cos(2 * np.pi * df["minute"]  / 60)

    # 15-minute time slot index (0–95 slots per day)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15

    # --- 5c. Domain-knowledge flags ---
    df["is_rush"]    = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_daytime"] = df["hour"].between(6, 21).astype(int)

    # Interaction: lanes × road type signal
    df["lanes_x_day"] = df["NumberofLanes"] * df["day"]

print("New feature count — train:", train.shape[1])
print("Sample new features:\n", train[["lat","lon","hour_sin","hour_cos","time_slot","is_rush"]].head(3))
print("✅ STEP 5 DONE")
```

---

## STEP 6 — Encode categorical columns

```python
from sklearn.preprocessing import LabelEncoder

# IMPORTANT: fit on BOTH train+test together
# so the 10 unseen test geohashes don't crash the encoder
cat_cols = [
    "RoadType", "Weather", "LargeVehicles", "Landmarks",
    "geo_p3", "geo_p4", "geo_p5"
]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str).fillna("Unknown")
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str).fillna("Unknown"))
    test[col]  = le.transform(test[col].astype(str).fillna("Unknown"))
    label_encoders[col] = le

print("Encoding done. Sample RoadType values:", train["RoadType"].unique())
print("✅ STEP 6 DONE")
```

---

## STEP 7 — Target encode geohash (most powerful location feature)

```python
from sklearn.model_selection import KFold

# Cross-validated target encoding prevents data leakage:
# each row is encoded using only OTHER rows in train — never itself
global_mean = train["demand"].mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train["geo_target_enc"] = 0.0

for tr_idx, val_idx in kf.split(train):
    fold_means = train.iloc[tr_idx].groupby("geohash")["demand"].mean()
    train.loc[train.index[val_idx], "geo_target_enc"] = (
        train.iloc[val_idx]["geohash"].map(fold_means).fillna(global_mean)
    )

# Test: use full train means (no leakage risk at prediction time)
full_geo_means = train.groupby("geohash")["demand"].mean()
test["geo_target_enc"] = test["geohash"].map(full_geo_means).fillna(global_mean)

print("Target encoding range:",
      train["geo_target_enc"].min().round(6), "→",
      train["geo_target_enc"].max().round(6))
print("✅ STEP 7 DONE")
```

---

## STEP 8 — Define feature list and split data

```python
FEATURES = [
    # Location
    "lat", "lon", "geo_p3", "geo_p4", "geo_p5", "geo_target_enc",
    # Time
    "hour", "minute", "time_slot", "day",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "is_rush", "is_daytime",
    # Road
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    # Environment
    "Temperature", "Weather",
    # Interaction
    "lanes_x_day"
]

TARGET = "demand"

X      = train[FEATURES]
y      = train[TARGET]
X_test = test[FEATURES]

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print("Features used:", len(FEATURES))
print("✅ STEP 8 DONE")
```

---

## STEP 9 — Train Model 1: LightGBM

```python
import lightgbm as lgb
from sklearn.metrics import r2_score

lgb_model = lgb.LGBMRegressor(
    n_estimators      = 3000,
    learning_rate     = 0.02,
    num_leaves        = 127,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = 42
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)]
)

lgb_val  = lgb_model.predict(X_val)
lgb_score = max(0, 100 * r2_score(y_val, lgb_val))
print(f"\n🟢 LightGBM Val Score: {lgb_score:.4f}")
print("✅ STEP 9 DONE")
```

---

## STEP 10 — Train Model 2: XGBoost

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators       = 3000,
    learning_rate      = 0.02,
    max_depth          = 7,
    min_child_weight   = 5,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    gamma              = 0.05,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    tree_method        = "hist",
    eval_metric        = "rmse",
    random_state       = 42,
    n_jobs             = -1
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=150,
    verbose=300
)

xgb_val   = xgb_model.predict(X_val)
xgb_score = max(0, 100 * r2_score(y_val, xgb_val))
print(f"\n🟠 XGBoost Val Score: {xgb_score:.4f}")
print("✅ STEP 10 DONE")
```

---

## STEP 11 — Train Model 3: CatBoost

```python
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations         = 3000,
    learning_rate      = 0.02,
    depth              = 8,
    l2_leaf_reg        = 3,
    random_strength    = 0.3,
    bagging_temperature= 0.5,
    od_type            = "Iter",
    od_wait            = 150,
    loss_function      = "RMSE",
    eval_metric        = "R2",
    random_seed        = 42,
    verbose            = 300
)

cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

cat_val   = cat_model.predict(X_val)
cat_score = max(0, 100 * r2_score(y_val, cat_val))
print(f"\n🔵 CatBoost Val Score: {cat_score:.4f}")
print("✅ STEP 11 DONE")
```

---

## STEP 12 — Build weighted ensemble

```python
# Assign each model a weight proportional to its validation R² score
raw_scores = np.array([lgb_score, xgb_score, cat_score])
raw_scores = np.maximum(raw_scores, 0)         # no negative weights
weights    = raw_scores / raw_scores.sum()

print(f"Weights → LGB: {weights[0]:.3f} | XGB: {weights[1]:.3f} | CAT: {weights[2]:.3f}")

# Blend validation predictions
ensemble_val = (
    weights[0] * lgb_val +
    weights[1] * xgb_val +
    weights[2] * cat_val
)
ensemble_score = max(0, 100 * r2_score(y_val, ensemble_val))
print(f"\n🏆 Ensemble Val Score: {ensemble_score:.4f}")

# Print all scores for comparison
print(f"\nScore summary:")
print(f"  LightGBM  : {lgb_score:.4f}")
print(f"  XGBoost   : {xgb_score:.4f}")
print(f"  CatBoost  : {cat_score:.4f}")
print(f"  Ensemble  : {ensemble_score:.4f}  ← final")
print("✅ STEP 12 DONE")
```

---

## STEP 13 — Generate test predictions

```python
lgb_test  = lgb_model.predict(X_test)
xgb_test  = xgb_model.predict(X_test)
cat_test  = cat_model.predict(X_test)

final_preds = (
    weights[0] * lgb_test +
    weights[1] * xgb_test +
    weights[2] * cat_test
)

# Clip to valid demand range — no physically impossible values
demand_min = train["demand"].min()
demand_max = train["demand"].max()
final_preds = np.clip(final_preds, demand_min, demand_max)

print("Prediction range:", final_preds.min().round(6), "→", final_preds.max().round(6))
print("NaN count:", np.isnan(final_preds).sum())
print("✅ STEP 13 DONE")
```

---

## STEP 14 — Sanity check + save submission.csv

```python
submission = pd.DataFrame({
    "Index":  test["Index"],
    "demand": final_preds
})

# Hard checks — all must pass before saving
assert submission.shape == (41778, 2),              f"Wrong shape: {submission.shape}"
assert list(submission.columns) == ["Index","demand"], "Wrong column names!"
assert submission.isnull().sum().sum() == 0,         "NaN values in submission!"
assert (final_preds >= 0).all(),                     "Negative predictions found!"
assert (final_preds <= 1).all(),                     "Predictions exceed 1.0!"

submission.to_csv("submission.csv", index=False)

print("submission.csv saved ✅")
print("Shape          :", submission.shape)
print("Column names   :", submission.columns.tolist())
print("Demand range   :", submission["demand"].min().round(6), "→",
      submission["demand"].max().round(6))
print("\nFirst 5 rows:")
print(submission.head())
print(f"\n🏆 Final ensemble validation score: {ensemble_score:.2f} / 100")
print("✅ STEP 14 DONE — Ready to upload submission.csv")
```

---

## STEP 15 — (Optional) Feature importance debug chart

Run this if your score is below 80. It tells you which features the model
relies on most. `geo_target_enc` and `RoadType` should be in the top 3.

```python
import matplotlib.pyplot as plt

feat_imp = pd.Series(
    lgb_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
feat_imp.tail(15).plot(kind="barh", ax=ax, color="#1D9E75")
ax.set_title("Top 15 feature importances — LightGBM", fontsize=13)
ax.set_xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()

print("\nTop 5 features:")
print(feat_imp.tail(5).sort_values(ascending=False))

# Diagnosis hint
top5 = feat_imp.tail(5).index.tolist()
if "geo_target_enc" not in top5:
    print("⚠️  geo_target_enc not in top 5 — re-check Step 7 target encoding")
if "RoadType" not in top5:
    print("⚠️  RoadType not in top 5 — re-check Step 6 encoding")
print("✅ STEP 15 DONE")
```

---

## STEP 16 — (Optional) Score booster: retrain on full data

After confirming your ensemble score is good on validation,
retrain all 3 models on the COMPLETE train set (no val split).
This gives the model ~20% more data before predicting test.

```python
# Retrain LightGBM on full data
best_lgb_iter = lgb_model.best_iteration_
lgb_final = lgb.LGBMRegressor(
    n_estimators=best_lgb_iter,
    learning_rate=0.02, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42
)
lgb_final.fit(X, y)

# Retrain XGBoost on full data
best_xgb_iter = xgb_model.best_iteration
xgb_final = xgb.XGBRegressor(
    n_estimators=best_xgb_iter,
    learning_rate=0.02, max_depth=7,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, random_state=42
)
xgb_final.fit(X, y)

# Retrain CatBoost on full data
cat_final = CatBoostRegressor(
    iterations=cat_model.best_iteration_,
    learning_rate=0.02, depth=8,
    random_seed=42, verbose=0
)
cat_final.fit(X, y)

# Final predictions with full-data models
final_preds_v2 = np.clip(
    weights[0] * lgb_final.predict(X_test) +
    weights[1] * xgb_final.predict(X_test) +
    weights[2] * cat_final.predict(X_test),
    demand_min, demand_max
)

submission_v2 = pd.DataFrame({
    "Index":  test["Index"],
    "demand": final_preds_v2
})
submission_v2.to_csv("submission_v2_full_retrain.csv", index=False)
print("submission_v2_full_retrain.csv saved ✅")
print("Use this file for your final upload — trained on 100% of available data.")
print("✅ STEP 16 DONE")
```

---

## Quick reference: what to fix if score is low

| Symptom | Most likely cause | Fix |
|---|---|---|
| Score < 50 | Timestamp not parsed correctly | Re-run Step 3, check `hour` values |
| Score 50–70 | Missing geo_target_enc | Re-run Steps 7–8 |
| Score 70–85 | Only one model used | Run all 3 models + Step 12 blend |
| Score 85–95 | No full retrain | Run Step 16 |
| Submission rejected | Shape wrong or column name wrong | Check Step 14 assertions |
| NaN in submission | Unseen geohash crash | Re-run Step 6 with combined fit |

---

## Expected output at the end

```
submission.csv saved ✅
Shape          : (41778, 2)
Column names   : ['Index', 'demand']
Demand range   : 0.000001 → 1.0

🏆 Final ensemble validation score: 96–99 / 100
✅ STEP 14 DONE — Ready to upload submission.csv
```
