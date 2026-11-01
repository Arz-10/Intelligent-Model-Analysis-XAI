"""
scratch/test_iloc_fix.py
------------------------
Targeted regression test for the KeyError: 0 fix.

Verifies that positional row selection via .iloc works correctly on
prep_res.X_test and prep_res.y_test, and that downstream SHAP / LIME
functions accept a pandas Series without error.
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np

# Ensure project root is on the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_preprocessing_artifacts
from src.model_trainer  import load_all_trained_models
from src.explainability import (
    get_tree_explainer,
    get_lime_explainer,
    compute_instance_shap,
    compute_instance_lime,
)

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label, fn):
    try:
        fn()
        results.append((PASS, label))
        print(f"  [PASS] {label}")
    except Exception as exc:
        results.append((FAIL, label))
        print(f"  [FAIL] {label}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
print("\nLoading preprocessing artifacts ...")
prep_res = load_preprocessing_artifacts()
if prep_res is None:
    print("ERROR: load_preprocessing_artifacts() returned None. Run Phase 3 first.")
    sys.exit(1)

X_test = prep_res.X_test
y_test = prep_res.y_test
n_rows = len(X_test)
mid_idx = n_rows // 2

print(f"  X_test type  : {type(X_test).__name__}")
print(f"  X_test shape : {X_test.shape}")
print(f"  X_test index : [{X_test.index[0]} ... {X_test.index[-1]}]  (first and last label)")
print(f"  y_test type  : {type(y_test).__name__}")
print(f"  n_rows       : {n_rows}")
print(f"  mid_idx      : {mid_idx}")

# ---------------------------------------------------------------------------
# 1. Positional row selection with .iloc
# ---------------------------------------------------------------------------
print("\n--- 1. Positional row selection ---")


def _select_first():
    row = X_test.iloc[0]
    assert isinstance(row, pd.Series), f"Expected Series, got {type(row)}"


def _select_middle():
    row = X_test.iloc[mid_idx]
    assert isinstance(row, pd.Series), f"Expected Series, got {type(row)}"


def _select_last():
    row = X_test.iloc[n_rows - 1]
    assert isinstance(row, pd.Series), f"Expected Series, got {type(row)}"


def _select_y_first():
    val = int(y_test.iloc[0])
    assert val in (0, 1), f"Unexpected y value: {val}"


def _select_y_last():
    val = int(y_test.iloc[n_rows - 1])
    assert val in (0, 1), f"Unexpected y value: {val}"


check("iloc[0] returns pd.Series", _select_first)
check(f"iloc[{mid_idx}] returns pd.Series", _select_middle)
check(f"iloc[{n_rows-1}] returns pd.Series", _select_last)
check("y_test.iloc[0] is binary int", _select_y_first)
check(f"y_test.iloc[{n_rows-1}] is binary int", _select_y_last)

# ---------------------------------------------------------------------------
# 2. Old label-based access raises KeyError
# ---------------------------------------------------------------------------
print("\n--- 2. Confirm old syntax raises KeyError ---")


def _old_syntax_raises():
    try:
        _ = X_test[0]   # label-based — should raise if index is not 0-based
        # If the index happens to contain 0 this would succeed;
        # the important thing is .iloc is the correct fix.
        print("    (note: X_test[0] did not raise — index may start at 0 in this split)")
    except KeyError:
        print("    (confirmed: X_test[0] raises KeyError as expected)")


_old_syntax_raises()

# ---------------------------------------------------------------------------
# 3. SHAP accepts pd.Series from .iloc
# ---------------------------------------------------------------------------
print("\n--- 3. SHAP instance explanation ---")

models = load_all_trained_models()
rf_model = models.get("Random Forest")
xgb_model = models.get("XGBoost")

if rf_model is None:
    print("  SKIP: random_forest model not found — run Phase 4 first")
else:
    rf_explainer = get_tree_explainer(rf_model)

    def _shap_first():
        x_row = X_test.iloc[0]
        result = compute_instance_shap(
            rf_model, x_row, prep_res.feature_names, explainer=rf_explainer
        )
        assert "predicted_prob" in result, f"Missing 'predicted_prob' key: {result.keys()}"
        assert "all_contributions" in result, f"Missing 'all_contributions' key: {result.keys()}"

    def _shap_last():
        x_row = X_test.iloc[n_rows - 1]
        result = compute_instance_shap(
            rf_model, x_row, prep_res.feature_names, explainer=rf_explainer
        )
        assert "predicted_prob" in result and "all_contributions" in result

    check("compute_instance_shap with iloc[0] (RF)", _shap_first)
    check(f"compute_instance_shap with iloc[{n_rows-1}] (RF)", _shap_last)

# ---------------------------------------------------------------------------
# 4. LIME accepts pd.Series from .iloc
# ---------------------------------------------------------------------------
print("\n--- 4. LIME instance explanation ---")

if rf_model is None:
    print("  SKIP: random_forest model not found")
else:
    lime_explainer = get_lime_explainer(prep_res.X_train, prep_res.feature_names)

    def _lime_first():
        x_row = X_test.iloc[0]
        result = compute_instance_lime(
            lime_explainer, rf_model.predict_proba, x_row, num_features=5
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "rules" in result, f"Missing 'rules' key: {result.keys()}"
        assert isinstance(result["rules"], list), "rules should be a list"
        assert len(result["rules"]) > 0, "LIME returned empty rules list"

    def _lime_mid():
        x_row = X_test.iloc[mid_idx]
        result = compute_instance_lime(
            lime_explainer, rf_model.predict_proba, x_row, num_features=5
        )
        assert isinstance(result, dict) and "rules" in result and len(result["rules"]) > 0

    check("compute_instance_lime with iloc[0] (RF)", _lime_first)
    check(f"compute_instance_lime with iloc[{mid_idx}] (RF)", _lime_mid)

# ---------------------------------------------------------------------------
# 5. Bounds check
# ---------------------------------------------------------------------------
print("\n--- 5. Bounds ---")


def _bounds_ok():
    for idx in [0, mid_idx, n_rows - 1]:
        assert 0 <= idx < n_rows, f"Index {idx} out of bounds for n_rows={n_rows}"


check("Sample indices within bounds", _bounds_ok)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
total = len(results)

print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed, {failed} failed")
print("="*50)

if failed:
    print("\nFailed tests:")
    for status, label in results:
        if status == FAIL:
            print(f"  - {label}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)

