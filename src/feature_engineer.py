"""
Steps 3–8 — Timestamp parsing, missing-value imputation, feature engineering,
categorical encoding, target encoding, and feature-list definition.
"""
import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# ── Step 3: Parse timestamp ──────────────────────────────────────────────────
def parse_timestamps(train, test):
    """Parse H:MM string timestamps into hour and minute integers."""
    def _parse(ts):
        parts = str(ts).split(":")
        return int(parts[0]), int(parts[1])

    for df in [train, test]:
        df[["hour", "minute"]] = df["timestamp"].apply(
            lambda x: pd.Series(_parse(x))
        )

    print("Train hours:", sorted(train["hour"].unique()))
    print("Test hours: ", sorted(test["hour"].unique()))
    return train, test


# ── Step 4: Fill missing values ──────────────────────────────────────────────
def fill_missing(train, test):
    """Impute RoadType (mode), Weather (mode), Temperature (geohash median)."""
    for df in [train, test]:
        df["RoadType"] = df["RoadType"].fillna(df["RoadType"].mode()[0])
        df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])

    # Temperature: smarter fill using location median
    temp_median_by_geo = train.groupby("geohash")["Temperature"].median()

    for df in [train, test]:
        df["Temperature"] = df["Temperature"].fillna(
            df["geohash"].map(temp_median_by_geo)
        )
        df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

    print("Missing after fill — train:", train.isnull().sum().sum())
    print("Missing after fill — test: ", test.isnull().sum().sum())
    return train, test


# ── Step 5: Feature engineering ──────────────────────────────────────────────
def engineer_features(train, test):
    """Create spatial, temporal, and interaction features."""
    def _decode_geo(h):
        try:
            result = pgh.decode(str(h))
            return float(result.latitude), float(result.longitude)
        except Exception:
            return 0.0, 0.0

    for df in [train, test]:
        coords = df["geohash"].apply(lambda x: pd.Series(_decode_geo(x)))
        df["lat"] = coords[0]
        df["lon"] = coords[1]

        # Geohash precision levels (zoom hierarchy)
        df["geo_p3"] = df["geohash"].str[:3]  # city-level zone
        df["geo_p4"] = df["geohash"].str[:4]  # district-level ~40km
        df["geo_p5"] = df["geohash"].str[:5]  # neighbourhood ~5km

        # Cyclical time encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        # 15-minute time slot index (0–95 slots per day)
        df["time_slot"] = df["hour"] * 4 + df["minute"] // 15

        # Domain-knowledge flags
        df["is_rush"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["is_daytime"] = df["hour"].between(6, 21).astype(int)

        # Interaction: lanes × day
        df["lanes_x_day"] = df["NumberofLanes"] * df["day"]

    print("New feature count — train:", train.shape[1])
    print("Sample new features:\n",
          train[["lat", "lon", "hour_sin", "hour_cos", "time_slot", "is_rush"]].head(3))
    return train, test


# ── Step 6: Encode categorical columns ──────────────────────────────────────
def encode_categoricals(train, test):
    """Label-encode categoricals fitted on combined train+test."""
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
        test[col] = le.transform(test[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le

    print("Encoding done. Sample RoadType values:", train["RoadType"].unique())
    return train, test, label_encoders


# ── Step 7: Target encode geohash ────────────────────────────────────────────
def target_encode_geohash(train, test):
    """Cross-validated target encoding of geohash to prevent leakage."""
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
    return train, test


# ── Step 8: Define feature list ──────────────────────────────────────────────
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


def get_feature_matrices(train, test):
    """Extract X, y, X_test from processed DataFrames."""
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]
    print(f"Feature matrix — Train: {X.shape} | Test: {X_test.shape}")
    print("Features used:", len(FEATURES))
    return X, y, X_test


# ── Master pipeline ─────────────────────────────────────────────────────────
def run_feature_pipeline(train, test):
    """Execute the full feature engineering pipeline (Steps 3–8)."""
    train, test = parse_timestamps(train, test)
    print("✅ STEP 3 DONE\n")

    train, test = fill_missing(train, test)
    print("✅ STEP 4 DONE\n")

    train, test = engineer_features(train, test)
    print("✅ STEP 5 DONE\n")

    train, test, label_encoders = encode_categoricals(train, test)
    print("✅ STEP 6 DONE\n")

    train, test = target_encode_geohash(train, test)
    print("✅ STEP 7 DONE\n")

    X, y, X_test = get_feature_matrices(train, test)
    print("✅ STEP 8 DONE\n")

    return train, test, X, y, X_test
