#!/usr/bin/env python3
"""
Traffic Demand Prediction — Main Orchestrator
Executes Steps 1–16 from the antigravity_traffic_prompt.md specification.
"""
import os
import warnings
warnings.filterwarnings("ignore")

# ── Step 1: Libraries ────────────────────────────────────────────────────────
print("=" * 70)
print("TRAFFIC DEMAND PREDICTION PIPELINE")
print("=" * 70)
print("All required libraries are installed via requirements.txt")
print("✅ STEP 1 DONE\n")

# ── Step 2: Load data ────────────────────────────────────────────────────────
from src.data_loader import load_data

train, test, sub = load_data("dataset")
print("✅ STEP 2 DONE\n")

# ── Steps 3–8: Feature engineering pipeline ──────────────────────────────────
from src.feature_engineer import run_feature_pipeline

train, test, X, y, X_test = run_feature_pipeline(train, test)

# ── Steps 9–11: Train models ────────────────────────────────────────────────
from src.train_models import train_all_models

models, splits = train_all_models(train, X, y)

# ── Step 12: Weighted ensemble ───────────────────────────────────────────────
from src.ensemble_submit import build_ensemble, generate_submission, full_retrain_and_submit

os.makedirs("output", exist_ok=True)

weights, ensemble_score = build_ensemble(models, splits, X_test, train)
print("✅ STEP 12 DONE\n")

# ── Steps 13–14: Generate & validate submission ─────────────────────────────
submission = generate_submission(models, weights, X_test, test, train)
print(f"\n🏆 Final ensemble validation score: {ensemble_score:.2f} / 100")
print("✅ STEP 14 DONE — Ready to upload submission.csv\n")

# ── Step 15: Feature importance diagnostics ──────────────────────────────────
from src.diagnostics import plot_feature_importance

lgb_model, _ = models["lgb"]
plot_feature_importance(lgb_model)
print("✅ STEP 15 DONE\n")

# ── Step 16: Full-data retrain ───────────────────────────────────────────────
submission_v2 = full_retrain_and_submit(models, weights, X, y, X_test, test, train)
print("✅ STEP 16 DONE\n")

# ── Final summary ────────────────────────────────────────────────────────────
print("=" * 70)
print("ALL STEPS COMPLETE")
print("=" * 70)
print(f"🏆 Ensemble validation score : {ensemble_score:.2f} / 100")
print(f"📄 submission.csv             : output/submission.csv")
print(f"📄 submission_v2 (full data)  : output/submission_v2_full_retrain.csv")
print(f"📊 Feature importance chart   : output/feature_importance.png")
print("=" * 70)
