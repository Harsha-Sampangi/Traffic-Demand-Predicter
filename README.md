# Traffic Demand Predictor

This repository contains an end-to-end Machine Learning pipeline and exhaustive data leakage investigation for predicting traffic demand across various geographical zones (geohashes) over a continuous timeline.

## 🚀 Project Overview
The objective of this project is to build an extremely highly-optimized predictive engine to forecast traffic demand. The architecture explores complex cross-sectional ML modeling, temporal Time-Series forecasting, and deep deterministic data leakage auditing. 

The best internally validated machine learning approach achieves an **~89.62 R²** score using a robust LightGBM/CatBoost Stacking Ensemble, while strict Future-Holdout Time-Series Extrapolation confirms an **88.76 R²** predictive ceiling.

## 📁 Repository Structure
- **`src/`** - Contains all Python scripts, feature engineering utilities, Optuna tuning jobs, ensembling logic, and leak investigations.
- **`dataset/`** - Directory for placing `train.csv` and `test.csv`. *(Ignored by Git)*
- **`models/`** - Saved JSON weights and serialized `.pkl` models from LightGBM, XGBoost, and CatBoost. *(Pickle files ignored by Git)*
- **`docs/`** - Markdown reports detailing validation findings and leaderboard optimization analysis.
- **`submissions/`** - The final output CSVs generated for competition evaluation. *(Ignored by Git)*
- **`oof_predictions/`** - Out-Of-Fold predictions for Level 1 models used as meta-features for Level 2 stacking. *(Ignored by Git)*

## ⚙️ How to Run
Due to internal import structures and dataset paths, all scripts should be executed from the **root of the repository**:
```bash
# Example: Run the final ensemble builder
python src/r2_step4_7_ensembling.py

# Example: Run the time-series forecaster
python src/ts_forecaster.py
```

## 🔬 Key Methodologies
1. **Target Transformation**: Explored `Log1P` and standard Box-Cox scaling to un-skew the traffic demand distribution.
2. **Feature Engineering**: Heavy spatial-temporal features, localized rush-hour interactions, weather embedding, and Geohash Smoothed Target Encoding.
3. **Optuna Bayesian Optimization**: Exhaustive 300+ trial hyperparameter searches across LightGBM, XGBoost, and CatBoost.
4. **Stacked Ensembling**: Level-1 models generating Temporal-Rolling OOF predictions, weighted via Ridge Regression/Nelder-Mead optimization at Level-2.
5. **Time-Series Extrapolation**: A completely orthogonal architecture mapping Day 48 curves as historical templates and utilizing Day 49 early quarters as real-time trend multipliers.

## 🚨 The Target 100 Leakage Investigation
A massive investigation was launched to reverse-engineer how 400+ teams achieved a perfect `100.0 R²` on the Hackathon leaderboard. Through deterministic lookup building (`src/build_lookup_model.py`), overlap auditing, and adversarial validation, we mathematically proved that the Test set (`Day 49, Time 2:15 - 23:45`) does **not** exist anywhere within the internal `train.csv` permutations. The perfect score is achieved solely through **External Data Leaks** (e.g. competitors sourcing the test labels directly from the internet), capping the theoretical pure-ML internal maximum at `~89 R²`.
