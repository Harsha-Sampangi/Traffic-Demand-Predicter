# Traffic Demand Predictor

An end-to-end Machine Learning pipeline for predicting traffic demand across geographical zones using a 3-model stacked ensemble (LightGBM + XGBoost + CatBoost).

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data files in dataset/
#    - train.csv (77,299 rows × 11 cols)
#    - test.csv  (41,778 rows × 10 cols)
#    - sample_submission.csv

# 3. Run the full pipeline
python main.py
```

## 📁 Project Structure

```
Traffic-Demand-Predicter/
├── main.py                     # Orchestrator — run this
├── requirements.txt            # Dependencies
├── antigravity_traffic_prompt.md  # Original problem specification
├── .gitignore
├── dataset/                    # Place CSV data here (git-ignored)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/                        # Source modules
│   ├── data_loader.py          # Step 2: Load & validate data
│   ├── feature_engineer.py     # Steps 3–8: Feature engineering
│   ├── train_models.py         # Steps 9–11: Model training
│   ├── ensemble_submit.py      # Steps 12–14, 16: Ensemble & submission
│   └── diagnostics.py          # Step 15: Feature importance
└── output/                     # Generated outputs (git-ignored)
    ├── submission.csv
    ├── submission_v2_full_retrain.csv
    └── feature_importance.png
```

## 📊 Scoring

```
score = max(0, 100 × R²_score(actual, predicted))
```

- Perfect predictions = **100**
- Worse than guessing the mean = **0**

## ⚙️ Pipeline Steps

| Step | Description | Module |
|------|-------------|--------|
| 1 | Install libraries | `requirements.txt` |
| 2 | Load & validate data shapes | `data_loader.py` |
| 3 | Parse `H:MM` timestamps | `feature_engineer.py` |
| 4 | Fill missing values | `feature_engineer.py` |
| 5 | Engineer features (geohash, cyclical time, rush-hour) | `feature_engineer.py` |
| 6 | Label-encode categoricals | `feature_engineer.py` |
| 7 | Cross-validated target encode geohash | `feature_engineer.py` |
| 8 | Define 23-feature matrix | `feature_engineer.py` |
| 9 | Train LightGBM | `train_models.py` |
| 10 | Train XGBoost | `train_models.py` |
| 11 | Train CatBoost | `train_models.py` |
| 12 | Build weighted ensemble | `ensemble_submit.py` |
| 13 | Generate test predictions | `ensemble_submit.py` |
| 14 | Sanity check & save submission | `ensemble_submit.py` |
| 15 | Feature importance diagnostics | `diagnostics.py` |
| 16 | Full-data retrain & final submission | `ensemble_submit.py` |

## 🔍 Validation Strategy

- **Primary**: Day 48 → Day 49 temporal split (no future leakage)
- **Secondary**: 80/20 random split (as baseline sanity check)

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Score < 50 | Timestamp parsing error | Check `hour` values after Step 3 |
| Score 50–70 | Missing geo_target_enc | Re-check Step 7 |
| Score 70–85 | Single model used | Ensure all 3 models + ensemble |
| Score 85–95 | No full retrain | Run Step 16 |
| Submission rejected | Shape/column error | Check Step 14 assertions |
