"""
scratch/test_phase6.py
----------------------
Unit and integration tests for Phase 6 (Explainability & Prediction Explorer).
Validates SHAP, LIME, cache persistence, instance explanations, and ensemble explanations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import joblib
import numpy as np
import pandas as pd

from src.explainability import (
    get_status,
    load_global_shap_cache,
    save_global_shap_cache,
    get_tree_explainer,
    get_lime_explainer,
    compute_instance_shap,
    compute_instance_lime,
    compute_instance_ensemble_shap,
    compute_global_shap_importance,
    get_domain_context,
)


def test_status():
    print("[Test 1] Testing get_status()...")
    status = get_status()
    assert status["shap_available"], "SHAP should be available in .venv"
    assert status["lime_available"], "LIME should be available in .venv"
    assert status["cache_exists"], "Global SHAP cache file should exist on disk"
    assert "random_forest" in status["cached_models"], "random_forest should be cached"
    assert "xgboost" in status["cached_models"], "xgboost should be cached"
    assert "weighted_soft_voting" in status["cached_models"], "weighted_soft_voting should be cached"
    print("  -> PASS: get_status() returns valid status and detected caches.")


def test_cache_structure():
    print("[Test 2] Testing load_global_shap_cache() and data structure...")
    cache = load_global_shap_cache()
    assert cache is not None, "Cache should load successfully"
    for m in ["random_forest", "xgboost", "weighted_soft_voting"]:
        assert m in cache, f"{m} missing from cache"
        ranking = cache[m]["ranking"]
        assert len(ranking) == 36, f"{m} ranking should contain 36 features, got {len(ranking)}"
        assert ranking[0]["rank"] == 1, "First feature should have rank 1"
        assert "mean_abs_shap" in ranking[0], "Ranking must include mean_abs_shap"
        assert "relative_importance_pct" in ranking[0], "Ranking must include relative_importance_pct"
        assert "impact_direction" in ranking[0], "Ranking must include impact_direction"
        # Summary distribution
        summary = cache[m]["summary_distribution"]
        assert len(summary) >= 10, f"Summary distribution should have at least 10 items, got {len(summary)}"
        assert "shap_values" in summary[0], "Summary item must have shap_values"
        assert "normalized_values" in summary[0], "Summary item must have normalized_values"
    print(f"  -> PASS: Cache validated for all models. Top RF feature: {cache['random_forest']['ranking'][0]['feature']}")


def test_single_instance_shap():
    print("[Test 3] Testing compute_instance_shap() on Random Forest...")
    rf = joblib.load("models/random_forest.pkl")
    X_test = np.load("outputs/X_test.npy")
    with open("outputs/preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names_encoded"]

    explainer = get_tree_explainer(rf)
    res = compute_instance_shap(rf, X_test[0], feature_names, explainer=explainer, top_k=5)

    assert "base_value" in res, "Missing base_value"
    assert "predicted_prob" in res, "Missing predicted_prob"
    assert 0.0 <= res["predicted_prob"] <= 1.0, f"Invalid predicted_prob: {res['predicted_prob']}"
    assert len(res["top_positive"]) <= 5, "top_positive exceeds top_k"
    assert len(res["top_negative"]) <= 5, "top_negative exceeds top_k"
    assert len(res["all_contributions"]) == 36, "all_contributions should have 36 items"

    print(f"  -> PASS: Instance SHAP computed. Prob: {res['predicted_prob']}, Base: {res['base_value']}")
    print(f"     Top positive factor: {res['top_positive'][0]['feature']} (SHAP: +{res['top_positive'][0]['shap_value']:.4f})")


def test_single_instance_lime():
    print("[Test 4] Testing compute_instance_lime() on Random Forest...")
    rf = joblib.load("models/random_forest.pkl")
    X_train = np.load("outputs/X_train.npy")
    X_test = np.load("outputs/X_test.npy")
    with open("outputs/preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names_encoded"]

    explainer = get_lime_explainer(X_train[:150], feature_names)
    res = compute_instance_lime(explainer, rf.predict_proba, X_test[0], num_features=6)

    assert "prediction_prob_late" in res, "Missing prediction_prob_late"
    assert len(res["rules"]) <= 6, "Exceeded requested num_features"
    assert "rule" in res["rules"][0], "Rule missing"
    assert "weight" in res["rules"][0], "Weight missing"
    assert "interpretation" in res["rules"][0], "Interpretation missing"

    print(f"  -> PASS: LIME computed. Prob Late: {res['prediction_prob_late']}")
    print(f"     Top LIME rule: {res['rules'][0]['rule']} (Weight: {res['rules'][0]['weight']:+.4f})")


def test_ensemble_shap():
    print("[Test 5] Testing compute_instance_ensemble_shap() on Weighted Soft Voting...")
    rf = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    X_test = np.load("outputs/X_test.npy")
    with open("outputs/preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names_encoded"]

    weights = {"Random Forest": 0.70, "XGBoost": 0.30}
    res = compute_instance_ensemble_shap(rf, xgb, weights, X_test[0], feature_names, top_k=5)

    assert "ensemble_name" in res, "Missing ensemble_name"
    assert "predicted_prob" in res, "Missing predicted_prob"
    assert "rf_probability" in res, "Missing rf_probability"
    assert "xgb_probability" in res, "Missing xgb_probability"
    # Verify linear combination of probabilities
    expected_prob = round(0.70 * res["rf_probability"] + 0.30 * res["xgb_probability"], 4)
    assert abs(res["predicted_prob"] - expected_prob) < 0.01, "Ensemble prob should be 0.70*RF + 0.30*XGB"
    assert len(res["all_contributions"]) == 36, "Should have 36 contributions"

    print(f"  -> PASS: Ensemble SHAP verified. Ens Prob: {res['predicted_prob']} (RF: {res['rf_probability']}, XGB: {res['xgb_probability']})")
    print(f"     Top combined driver: {res['top_positive'][0]['feature']} (SHAP: +{res['top_positive'][0]['shap_value']:.4f})")


def test_domain_context():
    print("[Test 6] Testing get_domain_context()...")
    ctx1 = get_domain_context("Days for shipment (scheduled)")
    assert "transit time" in ctx1.lower() or "shipping" in ctx1.lower(), "Context should explain transit time"
    ctx2 = get_domain_context("Shipping Mode")
    assert "logistics" in ctx2.lower() or "class" in ctx2.lower(), "Context should explain shipping mode"
    print("  -> PASS: Domain context strings verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 6 VERIFICATION TEST SUITE")
    print("=" * 60)
    test_status()
    test_cache_structure()
    test_single_instance_shap()
    test_single_instance_lime()
    test_ensemble_shap()
    test_domain_context()
    print("=" * 60)
    print("ALL PHASE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

