"""
scratch/test_phase4.py
----------------------
Unit and integration tests for Phase 4: Model Training and Evaluation.
Tests synthetic training, metrics calculation, confusion matrix format,
ROC-AUC verification, model saving/loading, and recommendation logic.
"""

import sys, os, warnings
sys.path.insert(0, r'c:\Users\Aarushi Sontakke\Downloads\CIL_Project')

import numpy as np
import pandas as pd

from src.model_trainer import (
    build_models, train_single_model, train_all_models,
    save_model, load_model, load_all_trained_models,
    TrainingRecord, MODEL_FILENAMES,
)
from src.evaluator import (
    evaluate_model, evaluate_all_models, recommend_best_model,
    get_roc_curve_data, load_evaluation_results,
)


def make_synthetic_data(n_samples=300, n_features=10, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    # Simple linear decision boundary with noise
    prob = 1 / (1 + np.exp(-(X["feat_0"] * 1.5 + X["feat_1"] * 0.8)))
    y = pd.Series((prob > 0.5).astype(int), name="Late_delivery_risk")
    
    # Split 70/30
    split_idx = int(n_samples * 0.7)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def run_tests():
    passed = 0
    failed = 0

    def ok(test_name, detail=""):
        nonlocal passed
        passed += 1
        msg = f"  PASS  {test_name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def fail(test_name, detail=""):
        nonlocal failed
        failed += 1
        print(f"  FAIL  {test_name}  — {detail}")

    print("--- Starting Phase 4 Synthetic Unit Tests ---")

    # Test 1: Model Initialization
    try:
        models = build_models(class_ratio=1.0)
        assert len(models) == 5, f"Expected 5 models, got {len(models)}"
        expected_names = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "XGBoost",
        ]
        for name in expected_names:
            assert name in models, f"Missing model: {name}"
        ok("T01 Model Initialization", f"All 5 models initialized: {list(models.keys())}")
    except Exception as e:
        fail("T01 Model Initialization", str(e))

    # Test 2: Training flow on synthetic dataset
    X_tr, X_te, y_tr, y_te = make_synthetic_data()
    records = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Train each model
            for name, m in models.items():
                rec = train_single_model(name, m, X_tr, y_tr)
                assert rec.success, f"{name} training failed: {rec.error_message}"
                assert rec.train_time_s >= 0
                assert os.path.isfile(rec.artifact_path)
                records[name] = rec
        ok("T02 Synthetic Training Flow", f"Trained {len(records)} models successfully")
    except Exception as e:
        fail("T02 Synthetic Training Flow", str(e))

    # Test 3: Model Evaluation Metrics
    try:
        results_df, full_results = evaluate_all_models(records, X_te, y_te)
        assert len(results_df) == 5
        assert len(full_results) == 5

        for name, metrics in full_results.items():
            assert 0.0 <= metrics["accuracy"] <= 1.0, f"Invalid accuracy for {name}"
            assert 0.0 <= metrics["precision"] <= 1.0, f"Invalid precision for {name}"
            assert 0.0 <= metrics["recall"] <= 1.0, f"Invalid recall for {name}"
            assert 0.0 <= metrics["f1"] <= 1.0, f"Invalid F1 for {name}"
            assert metrics["roc_auc_available"], f"ROC-AUC not available for {name}"
            assert 0.0 <= metrics["roc_auc"] <= 1.0, f"Invalid ROC-AUC for {name}"

        ok("T03 Metrics Calculation", "Accuracy, Precision, Recall, F1, and ROC-AUC in [0, 1]")
    except Exception as e:
        fail("T03 Metrics Calculation", str(e))

    # Test 4: Confusion Matrix Format and Orientation
    try:
        for name, metrics in full_results.items():
            cm = np.array(metrics["confusion_matrix"])
            assert cm.shape == (2, 2), f"Expected (2, 2) confusion matrix for {name}"
            assert cm.sum() == len(y_te), f"CM sum {cm.sum()} != test set size {len(y_te)}"
            # labels are [0, 1] so:
            # cm[0,0] is TN (true=0, pred=0)
            # cm[0,1] is FP (true=0, pred=1)
            # cm[1,0] is FN (true=1, pred=0)
            # cm[1,1] is TP (true=1, pred=1)
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0
        ok("T04 Confusion Matrix Orientation", "Labels [0, 1] mapped to [[TN, FP], [FN, TP]]")
    except Exception as e:
        fail("T04 Confusion Matrix Orientation", str(e))

    # Test 5: ROC Curve Computation
    try:
        roc_data = get_roc_curve_data(records, X_te, y_te)
        assert len(roc_data) == 5, f"Expected 5 ROC curves, got {len(roc_data)}"
        for name, data in roc_data.items():
            assert "fpr" in data and "tpr" in data and "auc" in data
            assert len(data["fpr"]) == len(data["tpr"])
            assert 0.0 <= data["auc"] <= 1.0
        ok("T05 ROC Curve Generation", f"Generated curves for all {len(roc_data)} models")
    except Exception as e:
        fail("T05 ROC Curve Generation", str(e))

    # Test 6: Model Saving and Reloading
    try:
        loaded_models = load_all_trained_models()
        assert loaded_models is not None, "load_all_trained_models returned None"
        assert len(loaded_models) == 5, f"Expected 5 loaded models, got {len(loaded_models)}"
        for name in expected_names:
            assert name in loaded_models, f"Missing loaded model: {name}"
            # Verify prediction runs
            preds = loaded_models[name].predict(X_te)
            assert len(preds) == len(X_te)
        ok("T06 Model Saving & Reloading", "All 5 models saved, loaded, and verified on predict()")
    except Exception as e:
        fail("T06 Model Saving & Reloading", str(e))

    # Test 7: Evaluation Results Persistence
    try:
        saved_df, saved_full = load_evaluation_results()
        assert saved_df is not None and saved_full is not None
        assert len(saved_df) == 5
        assert len(saved_full) == 5
        assert list(saved_df["Model"]) == list(results_df["Model"])
        ok("T07 Evaluation Results Persistence", "model_comparison.csv and model_metrics.json reloaded")
    except Exception as e:
        fail("T07 Evaluation Results Persistence", str(e))

    # Test 8: Recommendation Logic
    try:
        best_name, explanation = recommend_best_model(results_df)
        assert best_name in expected_names, f"Unknown recommended model: {best_name}"
        assert "Recommended model" in explanation
        assert "ROC-AUC" in explanation
        ok("T08 Best Model Recommendation", f"Selected: {best_name}")
    except Exception as e:
        fail("T08 Best Model Recommendation", str(e))

    # Test 9: Handling of Model Training Failure Gracefully
    try:
        failed_records = records.copy()
        failed_records["Faulty Model"] = TrainingRecord(
            model_name="Faulty Model",
            success=False,
            model=None,
            error_message="Simulated hardware timeout",
            train_time_s=1.23,
        )
        f_df, f_full = evaluate_all_models(failed_records, X_te, y_te)
        faulty_row = f_df[f_df["Model"] == "Faulty Model"].iloc[0]
        assert faulty_row["Status"] == "Failed"
        assert pd.isna(faulty_row["Accuracy"])
        assert f_full["Faulty Model"]["error"] == "Simulated hardware timeout"
        ok("T09 Graceful Failure Handling", "Failed models reported with NaN metrics and error message")
    except Exception as e:
        fail("T09 Graceful Failure Handling", str(e))

    print()
    print(f"Results: {passed}/{passed + failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)

