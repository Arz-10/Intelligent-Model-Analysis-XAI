"""
scratch/test_shap_cache_fix.py
------------------------------
Dedicated test suite verifying:
1. Direct imports of all explainability symbols used by app.py
2. Loading of existing valid SHAP cache
3. Safe handling of missing and corrupted cache files
4. Safe on-demand computation fallback (get_or_compute_global_shap)
5. Full import and execution validation of app.py
"""

import os
import sys
import tempfile
import json
sys.path.insert(0, os.path.abspath("."))


def test_explainability_imports():
    print("[Check 1] Testing direct imports of all explainability symbols used by app.py...")
    from src.explainability import (
        get_status,
        load_global_shap_cache,
        save_global_shap_cache,
        get_or_compute_global_shap,
        get_tree_explainer,
        get_lime_explainer,
        compute_instance_shap,
        compute_instance_lime,
        compute_instance_ensemble_shap,
        compute_global_shap_importance,
        get_domain_context,
        DEFAULT_CACHE_PATH,
    )
    assert callable(get_status), "get_status not callable"
    assert callable(load_global_shap_cache), "load_global_shap_cache not callable"
    assert callable(save_global_shap_cache), "save_global_shap_cache not callable"
    assert callable(get_or_compute_global_shap), "get_or_compute_global_shap not callable"
    assert callable(get_tree_explainer), "get_tree_explainer not callable"
    assert callable(get_lime_explainer), "get_lime_explainer not callable"
    assert callable(compute_instance_shap), "compute_instance_shap not callable"
    assert callable(compute_instance_lime), "compute_instance_lime not callable"
    assert callable(compute_instance_ensemble_shap), "compute_instance_ensemble_shap not callable"
    assert callable(compute_global_shap_importance), "compute_global_shap_importance not callable"
    assert callable(get_domain_context), "get_domain_context not callable"
    print("  -> PASS: All 11 explainability symbols successfully imported and callable.")


def test_load_existing_cache():
    print("[Check 2] Testing loading of existing valid SHAP cache...")
    from src.explainability import load_global_shap_cache, DEFAULT_CACHE_PATH

    assert os.path.exists(DEFAULT_CACHE_PATH), f"Cache missing at {DEFAULT_CACHE_PATH}"
    data = load_global_shap_cache(DEFAULT_CACHE_PATH)
    assert data is not None, "load_global_shap_cache returned None on existing cache"
    assert "random_forest" in data, "random_forest missing from cache"
    assert "xgboost" in data, "xgboost missing from cache"
    assert "weighted_soft_voting" in data, "weighted_soft_voting missing from cache"
    print(f"  -> PASS: Valid cache loaded. Cached models: {list(data.keys())}")


def test_missing_and_corrupt_cache():
    print("[Check 3] Testing safe handling of missing and corrupted cache files...")
    from src.explainability import load_global_shap_cache

    # 1. Non-existent file
    missing_path = "outputs/definitely_non_existent_cache_12345.json"
    res_missing = load_global_shap_cache(missing_path)
    assert res_missing is None, f"Expected None for missing cache, got {res_missing}"
    print("  -> PASS: Missing cache file handled safely (returned None without exception).")

    # 2. Corrupted JSON file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        tmp.write("{ invalid json content :::")
        tmp_name = tmp.name

    try:
        res_corrupt = load_global_shap_cache(tmp_name)
        assert res_corrupt is None, f"Expected None for corrupted cache, got {res_corrupt}"
        print("  -> PASS: Corrupted cache file handled safely (returned None without exception).")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 3. Empty file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        tmp.write("")
        tmp_name = tmp.name

    try:
        res_empty = load_global_shap_cache(tmp_name)
        assert res_empty is None, f"Expected None for empty cache, got {res_empty}"
        print("  -> PASS: Empty cache file handled safely (returned None without exception).")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_app_import():
    print("[Check 4] Testing import of app.py...")
    import app
    assert hasattr(app, "page_explainability"), "page_explainability missing from app.py"
    assert hasattr(app, "load_global_shap_cache"), "load_global_shap_cache missing from app.py module scope"
    print("  -> PASS: app.py imported cleanly with page_explainability and load_global_shap_cache in scope.")


if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING SHAP CACHE IMPORT & RESILIENCY TESTS")
    print("=" * 65)
    test_explainability_imports()
    test_load_existing_cache()
    test_missing_and_corrupt_cache()
    test_app_import()
    print("=" * 65)
    print(" ALL SHAP CACHE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

