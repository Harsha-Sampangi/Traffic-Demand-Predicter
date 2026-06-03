"""
Step 15 — Feature importance diagnostics.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.feature_engineer import FEATURES


def plot_feature_importance(lgb_model, output_dir="output"):
    """Generate feature importance bar chart and diagnostic warnings."""
    feat_imp = pd.Series(
        lgb_model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    feat_imp.tail(15).plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title("Top 15 feature importances — LightGBM", fontsize=13)
    ax.set_xlabel("Importance score")
    plt.tight_layout()

    out_path = f"{output_dir}/feature_importance.png"
    plt.savefig(out_path, dpi=120)
    plt.close()

    print(f"\nFeature importance chart saved to {out_path}")
    print("\nTop 5 features:")
    print(feat_imp.tail(5).sort_values(ascending=False))

    # Diagnosis hints
    top5 = feat_imp.tail(5).index.tolist()
    if "geo_target_enc" not in top5:
        print("⚠️  geo_target_enc not in top 5 — re-check Step 7 target encoding")
    if "RoadType" not in top5:
        print("⚠️  RoadType not in top 5 — re-check Step 6 encoding")
