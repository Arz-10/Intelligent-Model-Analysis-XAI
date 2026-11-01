"""
scratch/test_runtime_fixes.py
-----------------------------
Comprehensive automated test suite for the three reported runtime issues:
1. Preprocessing artifact access (dot, dict-key, tuple index, unpacking)
2. Positional row selection (.iloc) on X_test/y_test and SHAP/LIME compatibility
3. SHAP cache loading (valid, missing, corrupt)
"""

import os
import sys
import tempfile
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_preprocessing_artifacts, PreprocessingResult
from src.model_trainer import load_all_trained_models
from src.explainability import (
    load_global_shap_cache,
    save_global_shap_cache,
    get_or_compute_global_shap,
    get_tree_explainer,
    get_lime_explainer,
    compute_instance_shap,
    compute_instance_lime,
    compute_instance_ensemble_shap,
    DEFAULT_CACHE_PATH,
)

PASS = "PASS"
FAIL = "FAIL"
results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name} {detail}")


def test_artifact_access():
    print("\n--- Testing Preprocessing Artifact Access ---")
    arts = load_preprocessing_artifacts()
    if arts is None:
        record("Artifact Load", FAIL, "load_preprocessing_artifacts() returned None")
        return

    # 1. Attribute access
    try:
        assert isinstance(arts.X_train, pd.DataFrame)
        assert isinstance(arts.X_test, pd.DataFrame)
        assert isinstance(arts.y_train, pd.Series)
        assert isinstance(arts.y_test, pd.Series)
        assert isinstance(arts.feature_names, list)
        record("Attribute Access (.X_train, .y_train, etc.)", PASS, f"X_test shape: {arts.X_test.shape}")
    except Exception as e:
        record("Attribute Access", FAIL, str(e))

    # 2. Dict key access
    try:
        assert isinstance(arts["X_train"], pd.DataFrame)
        assert isinstance(arts["X_test"], pd.DataFrame)
        assert isinstance(arts["y_train"], pd.Series)
        assert isinstance(arts["y_test"], pd.Series)
        assert isinstance(arts["metadata"], dict)
        record("Dict Key Access (['X_train'], ['y_train'], etc.)", PASS)
    except Exception as e:
        record("Dict Key Access", FAIL, str(e))

    # 3. Positional tuple indexing and unpacking
    try:
        assert isinstance(arts[0], pd.DataFrame)  # X_train
        assert isinstance(arts[1], pd.DataFrame)  # X_test
        assert isinstance(arts[2], pd.Series)     # y_train
        assert isinstance(arts[3], pd.Series)     # y_test
        X_tr, X_te, y_tr, y_te, f_names, pipe, meta, errs = arts
        assert len(f_names) == 36
        record("Tuple Indexing and Unpacking (arts[0], X_tr, ...)", PASS)
    except Exception as e:
        record("Tuple Indexing and Unpacking", FAIL, str(e))

    # 4. Dict helper methods (.get, in)
    try:
        assert arts.get("metadata") is not None
        assert arts.get("nonexistent_key", "default") == "default"
        assert "X_test" in arts
        assert "pipeline" in arts
        record("Dict Helper Methods (.get(), 'key' in arts)", PASS)
    except Exception as e:
        record("Dict Helper Methods", FAIL, str(e))


def test_row_selection_and_explainers():
    print("\n--- Testing Row Selection (.iloc) and Downstream Explainers ---")
    arts = load_preprocessing_artifacts()
    models = load_all_trained_models()
    rf_model = models.get("Random Forest")
    xgb_model = models.get("XGBoost")

    X_test = arts.X_test
    y_test = arts.y_test
    n_rows = len(X_test)
    test_indices = [0, n_rows // 2, n_rows - 1]

    # Row selection
    for idx in test_indices:
        try:
            x_row = X_test.iloc[idx]
            y_val = int(y_test.iloc[idx])
            assert isinstance(x_row, pd.Series)
            assert len(x_row) == 36
            assert y_val in (0, 1)
            record(f"Row Selection at index {idx}", PASS, f"y_true={y_val}")
        except Exception as e:
            record(f"Row Selection at index {idx}", FAIL, str(e))

    # SHAP Instance Explanation
    if rf_model is not None:
        try:
            expl = get_tree_explainer(rf_model)
            x_inst = X_test.iloc[0]
            shap_res = compute_instance_shap(rf_model, x_inst, arts.feature_names, explainer=expl)
            assert "predicted_prob" in shap_res
            assert "base_value" in shap_res
            assert "all_contributions" in shap_res
            assert len(shap_res["all_contributions"]) == 36
            record("SHAP compute_instance_shap with Series input", PASS, f"prob={shap_res['predicted_prob']:.4f}")
        except Exception as e:
            record("SHAP compute_instance_shap", FAIL, str(e))

    # LIME Instance Explanation
    if rf_model is not None:
        try:
            lime_expl = get_lime_explainer(arts.X_train[:100], arts.feature_names)
            x_inst = X_test.iloc[0]
            lime_res = compute_instance_lime(lime_expl, rf_model.predict_proba, x_inst, num_features=5)
            assert "rules" in lime_res
            assert len(lime_res["rules"]) > 0
            record("LIME compute_instance_lime with Series input", PASS, f"rules count={len(lime_res['rules'])}")
        except Exception as e:
            record("LIME compute_instance_lime", FAIL, str(e))

    # Ensemble SHAP Instance Explanation
    if rf_model is not None and xgb_model is not None:
        try:
            rf_expl = get_tree_explainer(rf_model)
            xgb_expl = get_tree_explainer(xgb_model)
            x_inst = X_test.iloc[0]
            ens_shap = compute_instance_ensemble_shap(
                rf_model, xgb_model, {"Random Forest": 0.70, "XGBoost": 0.30},
                x_inst, arts.feature_names,
                rf_explainer=rf_expl, xgb_explainer=xgb_expl,
            )
            assert "predicted_prob" in ens_shap
            assert len(ens_shap["all_contributions"]) == 36
            record("Ensemble SHAP compute_instance_ensemble_shap", PASS, f"ens_prob={ens_shap['predicted_prob']:.4f}")
        except Exception as e:
            record("Ensemble SHAP", FAIL, str(e))


def test_shap_cache_scenarios():
    print("\n--- Testing SHAP Cache Scenarios ---")

    # 1. Existing valid cache
    try:
        cache = load_global_shap_cache(DEFAULT_CACHE_PATH)
        assert cache is not None
        assert isinstance(cache, dict)
        assert len(cache) > 0
        first_model = list(cache.keys())[0]
        assert "ranking" in cache[first_model]
        record("SHAP Cache - Valid File Loading", PASS, f"Models: {list(cache.keys())}")
    except Exception as e:
        record("SHAP Cache - Valid File Loading", FAIL, str(e))

    # 2. Missing cache file
    try:
        missing_path = os.path.join("outputs", "non_existent_cache_test.json")
        res = load_global_shap_cache(missing_path)
        assert res is None
        record("SHAP Cache - Missing File Handling", PASS, "Returned None gracefully")
    except Exception as e:
        record("SHAP Cache - Missing File Handling", FAIL, str(e))

    # 3. Invalid / malformed cache file
    temp_dir = tempfile.gettempdir()
    corrupt_path = os.path.join(temp_dir, "corrupt_shap_cache_test.json")
    try:
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json content: true,,, ")
        res = load_global_shap_cache(corrupt_path)
        assert res is None
        record("SHAP Cache - Malformed File Handling", PASS, "Returned None gracefully without exception")
    except Exception as e:
        record("SHAP Cache - Malformed File Handling", FAIL, str(e))
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)

    # 4. Empty JSON object
    empty_path = os.path.join(temp_dir, "empty_shap_cache_test.json")
    try:
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("{}")
        res = load_global_shap_cache(empty_path)
        assert res is None
        record("SHAP Cache - Empty JSON Handling", PASS, "Returned None gracefully for empty cache")
    except Exception as e:
        record("SHAP Cache - Empty JSON Handling", FAIL, str(e))
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)


def main():
    test_artifact_access()
    test_row_selection_and_explainers()
    test_shap_cache_scenarios()

    print("\n==========================================")
    print("SUMMARY")
    print("==========================================")
    n_pass = sum(1 for _, status, _ in results if status == PASS)
    n_fail = sum(1 for _, status, _ in results if status == FAIL)
    print(f"Total: {len(results)} | Passed: {n_pass} | Failed: {n_fail}")

    if n_fail > 0:
        print("\nFailures:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("All runtime fix tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

