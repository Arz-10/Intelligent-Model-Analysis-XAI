import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import joblib
import numpy as np
from src.explainability import (
    compute_global_shap_importance,
    save_global_shap_cache,
    DEFAULT_CACHE_PATH,
)

def main():
    print("Loading models and preprocessed data...")
    rf = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    X_test = np.load("outputs/X_test.npy")

    with open("outputs/preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names_encoded"]

    # 150 representative samples for balanced speed and stability
    n_samples = 150
    sample = X_test[:n_samples]

    print(f"Computing Global SHAP for Random Forest ({n_samples} samples)...")
    t0 = time.time()
    rf_shap = compute_global_shap_importance(
        rf, sample, feature_names, model_name="random_forest"
    )
    print(f"Random Forest SHAP completed in {time.time() - t0:.2f}s.")

    print(f"Computing Global SHAP for XGBoost ({n_samples} samples)...")
    t0 = time.time()
    xgb_shap = compute_global_shap_importance(
        xgb, sample, feature_names, model_name="xgboost"
    )
    print(f"XGBoost SHAP completed in {time.time() - t0:.2f}s.")

    # Weighted ensemble combination (0.70 RF + 0.30 XGB)
    w_rf, w_xgb = 0.70, 0.30
    rf_map = {r["feature"]: r["mean_abs_shap"] for r in rf_shap["ranking"]}
    xgb_map = {r["feature"]: r["mean_abs_shap"] for r in xgb_shap["ranking"]}

    ens_ranking = []
    ens_scores = []
    for f in feature_names:
        score = w_rf * rf_map.get(f, 0.0) + w_xgb * xgb_map.get(f, 0.0)
        ens_scores.append((f, score))

    ens_scores.sort(key=lambda x: x[1], reverse=True)
    total_score = sum(s[1] for s in ens_scores) or 1e-9

    for rank, (feat, score) in enumerate(ens_scores, start=1):
        rel_pct = (score / total_score) * 100.0
        # inherit direction from dominant RF model
        direction = next((r["impact_direction"] for r in rf_shap["ranking"] if r["feature"] == feat), "Context-dependent")
        ens_ranking.append({
            "rank": rank,
            "feature": feat,
            "mean_abs_shap": round(score, 5),
            "relative_importance_pct": round(rel_pct, 2),
            "impact_direction": direction,
        })

    ens_shap = {
        "model_name": "weighted_soft_voting",
        "n_samples": n_samples,
        "ranking": ens_ranking,
        "summary_distribution": rf_shap["summary_distribution"][:15],
    }

    cache_data = {
        "random_forest": rf_shap,
        "xgboost": xgb_shap,
        "weighted_soft_voting": ens_shap,
        "metadata": {
            "n_samples": n_samples,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "DataCo Smart Supply Chain",
            "n_features": len(feature_names),
        }
    }

    path = save_global_shap_cache(cache_data)
    print(f"Successfully cached global SHAP to: {path}")

if __name__ == "__main__":
    main()
