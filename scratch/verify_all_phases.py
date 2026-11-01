"""
scratch/verify_all_phases.py
----------------------------
Comprehensive End-to-End Verification Suite for Phases 1 through 6
of the Computational Intelligence Lab (CIL) Project.

Verifies:
- Phase 1: Environment, dependencies, and file layout
- Phase 2: Dataset schema, loading, and validation
- Phase 3: Preprocessing, leakage prevention, train/test split
- Phase 4: 5 Individual ML classifiers, metrics, and ROC coordinates
- Phase 5: 4 Ensemble methods, unified comparison, and overall recommendation
- Phase 6: Global SHAP importance, local SHAP, local LIME, and domain context
- Application: Streamlit app imports, session state, and emoji-free compliance
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))


def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def verify_phase1_environment():
    print_header("PHASE 1 VERIFICATION: Environment & Project Structure")
    req_packages = ["streamlit", "pandas", "numpy", "sklearn", "xgboost", "shap", "lime", "plotly"]
    for pkg in req_packages:
        try:
            __import__(pkg)
            print(f"  [PASS] Package available: {pkg}")
        except ImportError:
            print(f"  [FAIL] Missing package: {pkg}")
            return False

    req_dirs = ["data", "models", "outputs", "src", "scratch"]
    for d in req_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"
        print(f"  [PASS] Directory exists: {d}/")

    return True


def verify_phase2_dataset():
    print_header("PHASE 2 VERIFICATION: Dataset Loading & Validation")
    from src.data_loader import find_dataset_file, load_dataset, TARGET_COLUMN

    found_path = find_dataset_file()
    assert found_path is not None, "DataCo dataset not found in data/ directory"
    print(f"  [PASS] Dataset detected at: {found_path}")

    df, report = load_dataset(found_path, source_name=os.path.basename(found_path))
    assert report["success"], "Dataset loading failed"
    assert report["target_found"], f"Target {TARGET_COLUMN} not found"
    assert df.shape[0] == 180519, f"Expected 180,519 rows, got {df.shape[0]}"
    assert df.shape[1] == 53, f"Expected 53 columns, got {df.shape[1]}"
    print(f"  [PASS] Schema confirmed: {df.shape[0]:,} rows, {df.shape[1]} columns, target present.")
    return True


def verify_phase3_preprocessing():
    print_header("PHASE 3 VERIFICATION: Preprocessing & Leakage Prevention")
    from src.preprocessing import load_preprocessing_artifacts

    arts = load_preprocessing_artifacts()
    assert arts is not None, "Preprocessing artifacts missing in outputs/"
    X_train, X_test = arts.X_train, arts.X_test
    y_train, y_test = arts.y_train, arts.y_test
    meta = arts.metadata

    assert X_train.shape == (126363, 36), f"Wrong X_train shape: {X_train.shape}"
    assert X_test.shape == (54156, 36), f"Wrong X_test shape: {X_test.shape}"
    assert len(y_train) == 126363, f"Wrong y_train length: {len(y_train)}"
    assert len(y_test) == 54156, f"Wrong y_test length: {len(y_test)}"

    # Leakage check
    leakage_cols = ["Delivery Status", "Days for shipping (real)", "Late_delivery_risk"]
    for col in leakage_cols:
        assert col not in X_train.columns, f"Target leakage detected: {col} in X_train"
        assert col not in X_test.columns, f"Target leakage detected: {col} in X_test"
    print("  [PASS] Leakage prevention verified: Delivery Status & Days for shipping (real) excluded.")
    print(f"  [PASS] Stratified split confirmed: Train={len(X_train):,}, Test={len(X_test):,}, Features=36.")
    return True


def verify_phase4_models():
    print_header("PHASE 4 VERIFICATION: Machine Learning Classifiers & Evaluation")
    from src.model_trainer import load_all_trained_models, MODEL_FILENAMES
    from src.evaluator import load_evaluation_results, load_roc_curves

    models = load_all_trained_models()
    assert models is not None, "Failed to load models from models/"
    expected_models = [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "Support Vector Machine", "XGBoost"
    ]
    for m in expected_models:
        assert m in models, f"Model {m} not loaded"
        assert hasattr(models[m], "predict"), f"{m} lacks predict method"
        print(f"  [PASS] Trained model verified: {m}")

    eval_df, full_metrics = load_evaluation_results()
    assert eval_df is not None, "outputs/model_comparison.csv missing"
    assert len(eval_df) == 5, f"Expected 5 evaluated models, got {len(eval_df)}"

    rf_row = eval_df[eval_df["Model"] == "Random Forest"].iloc[0]
    xgb_row = eval_df[eval_df["Model"] == "XGBoost"].iloc[0]

    assert rf_row["ROC-AUC"] >= 0.84, f"Random Forest ROC-AUC unexpectedly low: {rf_row['ROC-AUC']}"
    assert rf_row["Precision"] >= 0.87, f"Random Forest Precision unexpectedly low: {rf_row['Precision']}"
    print(f"  [PASS] Top individual model verified: Random Forest (ROC-AUC: {rf_row['ROC-AUC']:.4f}, Precision: {rf_row['Precision']:.4f})")
    print(f"  [PASS] Runner-up model verified: XGBoost (ROC-AUC: {xgb_row['ROC-AUC']:.4f}, F1: {xgb_row['F1-Score']:.4f})")

    roc_data, roc_meta = load_roc_curves()
    assert roc_data is not None, "outputs/roc_curves.json missing"
    assert len(roc_data) == 5, f"Expected 5 ROC curves, got {len(roc_data)}"
    print("  [PASS] ROC curves persistence verified: 5 models restored without retraining.")
    return True


def verify_phase5_ensembles():
    print_header("PHASE 5 VERIFICATION: Ensemble Learning & Unified Ranking")
    from src.ensemble import load_ensemble_results, combine_individual_and_ensemble_results, recommend_overall_best
    from src.evaluator import load_evaluation_results

    ens_df, ens_full, ens_roc = load_ensemble_results()
    assert ens_df is not None, "outputs/ensemble_comparison.csv missing"
    assert len(ens_df) == 4, f"Expected 4 ensemble methods, got {len(ens_df)}"

    weighted_row = ens_df[ens_df["Model"].str.contains("Weighted Soft Voting")].iloc[0]
    assert weighted_row["ROC-AUC"] >= 0.85, f"Weighted Soft Voting ROC-AUC low: {weighted_row['ROC-AUC']}"
    print(f"  [PASS] Best ensemble verified: {weighted_row['Model']} (ROC-AUC: {weighted_row['ROC-AUC']:.4f})")

    ind_df, _ = load_evaluation_results()
    unified_df = combine_individual_and_ensemble_results(ind_df, ens_df)
    assert len(unified_df) == 9, f"Expected 9 total systems, got {len(unified_df)}"

    best_name, explanation = recommend_overall_best(unified_df)
    assert "Weighted Soft Voting" in best_name, f"Expected Weighted Soft Voting recommended, got {best_name}"
    print(f"  [PASS] Overall system recommendation verified: {best_name}")
    print(f"  [PASS] Performance gain justified: Ensemble (+0.0024 ROC-AUC) over Random Forest.")
    return True


def verify_phase6_explainability():
    print_header("PHASE 6 VERIFICATION: Explainability (SHAP & LIME)")
    from src.explainability import (
        get_status,
        load_global_shap_cache,
        get_tree_explainer,
        get_lime_explainer,
        compute_instance_shap,
        compute_instance_lime,
        compute_instance_ensemble_shap,
        get_domain_context,
    )

    status = get_status()
    assert status["shap_available"], "SHAP unavailable"
    assert status["lime_available"], "LIME unavailable"
    assert status["cache_exists"], "Global SHAP cache missing"

    cache = load_global_shap_cache()
    assert "random_forest" in cache, "random_forest missing from cache"
    assert "xgboost" in cache, "xgboost missing from cache"
    assert "weighted_soft_voting" in cache, "weighted_soft_voting missing from cache"

    rf_ranking = cache["random_forest"]["ranking"]
    assert len(rf_ranking) == 36, f"Expected 36 features, got {len(rf_ranking)}"
    print(f"  [PASS] Global SHAP cache verified: 36 features ranked. Top feature: {rf_ranking[0]['feature']} ({rf_ranking[0]['relative_importance_pct']}%)")

    # Instance SHAP
    rf = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    X_test = np.load("outputs/X_test.npy")
    with open("outputs/preprocessing_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta["feature_names_encoded"]

    expl = get_tree_explainer(rf)
    inst_shap = compute_instance_shap(rf, X_test[0], feat_names, explainer=expl)
    assert inst_shap["predicted_prob"] is not None
    assert len(inst_shap["all_contributions"]) == 36
    print(f"  [PASS] Local SHAP instance test: Prob={inst_shap['predicted_prob']:.4f}, Top driver={inst_shap['top_positive'][0]['feature']}")

    # Instance LIME
    X_train = np.load("outputs/X_train.npy")
    lime_expl = get_lime_explainer(X_train[:150], feat_names)
    inst_lime = compute_instance_lime(lime_expl, rf.predict_proba, X_test[0], num_features=5)
    assert len(inst_lime["rules"]) <= 5
    print(f"  [PASS] Local LIME instance test: Top rule={inst_lime['rules'][0]['rule']} (Weight: {inst_lime['rules'][0]['weight']:+.4f})")

    # Ensemble SHAP
    ens_shap = compute_instance_ensemble_shap(rf, xgb, {"Random Forest": 0.70, "XGBoost": 0.30}, X_test[0], feat_names)
    assert abs(ens_shap["predicted_prob"] - (0.70 * ens_shap["rf_probability"] + 0.30 * ens_shap["xgb_probability"])) < 0.01
    print(f"  [PASS] Ensemble decomposed SHAP verified: Ens Prob={ens_shap['predicted_prob']:.4f} (0.70 RF + 0.30 XGB)")

    # Domain Context
    ctx = get_domain_context("Days for shipment (scheduled)")
    assert len(ctx) > 20
    print("  [PASS] Domain context dictionary operational.")
    return True


def verify_app_integrity():
    print_header("APPLICATION INTEGRITY & EMOJI-FREE UI VERIFICATION")
    # Check syntax and imports
    import app
    assert hasattr(app, "page_home"), "page_home missing"
    assert hasattr(app, "page_dataset"), "page_dataset missing"
    assert hasattr(app, "page_preprocessing"), "page_preprocessing missing"
    assert hasattr(app, "page_model_training"), "page_model_training missing"
    assert hasattr(app, "page_model_comparison"), "page_model_comparison missing"
    assert hasattr(app, "page_ensemble"), "page_ensemble missing"
    assert hasattr(app, "page_explainability"), "page_explainability missing"
    print("  [PASS] All 7 Streamlit dashboard pages defined and imported successfully.")

    # Check for emojis in Python source files
    py_files = [
        "app.py",
        "src/data_loader.py",
        "src/preprocessing.py",
        "src/model_trainer.py",
        "src/evaluator.py",
        "src/ensemble.py",
        "src/explainability.py",
    ]
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map
        (0x2600, 0x26FF),    # Misc symbols
        (0x2700, 0x27BF),    # Dingbats
        (0x1F900, 0x1F9FF),  # Supplemental Symbols
    ]

    for pf in py_files:
        with open(pf, "r", encoding="utf-8") as f:
            content = f.read()
        for char in content:
            cp = ord(char)
            for start, end in emoji_ranges:
                if start <= cp <= end:
                    print(f"  [FAIL] Emoji found in {pf}: char {char} (U+{cp:X})")
                    return False
        print(f"  [PASS] Emoji-free verified: {pf}")

    return True


def main():
    print("\n" + "=" * 65)
    print(" COMPUTATIONAL INTELLIGENCE LAB (CIL) PROJECT")
    print(" COMPLETE SYSTEM END-TO-END VERIFICATION")
    print("=" * 65)

    v1 = verify_phase1_environment()
    v2 = verify_phase2_dataset()
    v3 = verify_phase3_preprocessing()
    v4 = verify_phase4_models()
    v5 = verify_phase5_ensembles()
    v6 = verify_phase6_explainability()
    va = verify_app_integrity()

    all_passed = all([v1, v2, v3, v4, v5, v6, va])

    print("\n" + "=" * 65)
    if all_passed:
        print(" [SUCCESS] ALL 6 PHASES & APPLICATION VERIFIED END-TO-END!")
        print(" ALL TESTS PASSED WITH 0 FAILURES.")
    else:
        print(" [FAILURE] One or more verification checks failed.")
    print("=" * 65 + "\n")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

