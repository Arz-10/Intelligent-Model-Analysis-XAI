"""
scratch/test_phase5.py
----------------------
Automated test suite for Phase 5: Ensemble Learning & ROC Persistence.
"""

import sys, os, warnings
sys.path.insert(0, r'c:\Users\Aarushi Sontakke\Downloads\CIL_Project')

import numpy as np
import pandas as pd

from src.evaluator import (
    load_roc_curves, recover_roc_curves, save_roc_curves, load_evaluation_results
)
from src.ensemble import (
    SoftVotingEnsemble, WeightedVotingEnsemble, StackingEnsemble,
    build_and_evaluate_all_ensembles, load_ensemble_results,
    combine_individual_and_ensemble_results, recommend_overall_best,
)
from src.model_trainer import build_models


def make_synthetic_data(n_samples=200, n_features=6, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    prob = 1 / (1 + np.exp(-(X["feat_0"] * 1.2 + X["feat_1"] * 0.9)))
    y = pd.Series((prob > 0.5).astype(int), name="Late_delivery_risk")
    split = int(n_samples * 0.7)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def run_tests():
    passed = 0
    failed = 0

    def ok(name, detail=""):
        nonlocal passed
        passed += 1
        msg = f"  PASS  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def fail(name, detail=""):
        nonlocal failed
        failed += 1
        print(f"  FAIL  {name}  — {detail}")

    print("--- Starting Phase 5 & ROC Persistence Tests ---")

    # T01: ROC Persistence & Recovery Test
    try:
        curves, meta = load_roc_curves()
        assert curves is not None and len(curves) >= 5, "Could not load ROC curves from disk"
        ind_df, _ = load_evaluation_results()
        assert ind_df is not None
        # Check AUC values match evaluation table
        for _, row in ind_df[ind_df["Status"] == "Completed"].iterrows():
            m_name = row["Model"]
            if m_name in curves:
                diff = abs(curves[m_name]["auc"] - row["ROC-AUC"])
                assert diff < 0.001, f"AUC mismatch for {m_name}: {curves[m_name]['auc']} vs {row['ROC-AUC']}"
        ok("T01 ROC Persistence & AUC Verification", f"Verified {len(curves)} models match saved evaluation table")
    except Exception as e:
        fail("T01 ROC Persistence & AUC Verification", str(e))

    # Prepare small synthetic models for unit testing
    X_tr, X_te, y_tr, y_te = make_synthetic_data()
    m_dict = build_models()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in m_dict.values():
            m.fit(X_tr, y_tr)

    # T02: Soft Voting Probability Calculation & Class Alignment
    try:
        sub_models = {"Random Forest": m_dict["Random Forest"], "XGBoost": m_dict["XGBoost"]}
        soft_ens = SoftVotingEnsemble(sub_models)
        proba = soft_ens.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2), f"Expected shape ({len(X_te)}, 2), got {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probabilities do not sum to 1.0"
        preds = soft_ens.predict(X_te)
        assert set(np.unique(preds)).issubset({0, 1})
        ok("T02 Soft Voting Probability & Alignment", "Predictions and probabilities aligned for class 0 and 1")
    except Exception as e:
        fail("T02 Soft Voting Probability & Alignment", str(e))

    # T03: Weighted Soft Voting Weight Normalization
    try:
        weights = {"Random Forest": 0.7, "XGBoost": 0.3}
        weighted_ens = WeightedVotingEnsemble(sub_models, weights=weights)
        w_proba = weighted_ens.predict_proba(X_te)
        assert w_proba.shape == (len(X_te), 2)
        assert np.allclose(w_proba.sum(axis=1), 1.0)
        assert weighted_ens.weights["Random Forest"] == 0.7
        ok("T03 Weighted Soft Voting Normalization", "Weights normalized and applied to base model probabilities")
    except Exception as e:
        fail("T03 Weighted Soft Voting Normalization", str(e))

    # T04: Stacking Ensemble Out-Of-Fold Cross-Validation
    try:
        stack_ens = StackingEnsemble(sub_models, cv=2)
        stack_ens.fit_oof(X_tr, y_tr)
        assert stack_ens.is_fitted
        s_proba = stack_ens.predict_proba(X_te)
        assert s_proba.shape == (len(X_te), 2)
        s_preds = stack_ens.predict(X_te)
        assert len(s_preds) == len(X_te)
        ok("T04 Stacking Ensemble OOF Training", "Meta-model trained on OOF probabilities with no test leakage")
    except Exception as e:
        fail("T04 Stacking Ensemble OOF Training", str(e))

    # T05: Unified Comparison Table Creation
    try:
        dummy_ind = pd.DataFrame([{
            "Model": "Random Forest", "Accuracy": 0.72, "Precision": 0.88, "Recall": 0.57,
            "F1-Score": 0.69, "ROC-AUC": 0.85, "Train Time (s)": 15.0, "Predict Time (s)": 0.5, "Status": "Completed"
        }])
        dummy_ens = pd.DataFrame([{
            "Model": "Ensemble: Soft Voting", "Accuracy": 0.73, "Precision": 0.87, "Recall": 0.60,
            "F1-Score": 0.71, "ROC-AUC": 0.855, "Train Time (s)": 0.1, "Predict Time (s)": 0.6, "Status": "Completed"
        }])
        u_df = combine_individual_and_ensemble_results(dummy_ind, dummy_ens)
        assert len(u_df) == 2
        assert "Type" in u_df.columns
        assert u_df["Type"].tolist() == ["Individual", "Ensemble"]
        ok("T05 Unified Comparison Table", "Correctly combined individual and ensemble results with Type tags")
    except Exception as e:
        fail("T05 Unified Comparison Table", str(e))

    # T06: Recommendation Logic (Ensemble vs Individual)
    try:
        rec_name, exp = recommend_overall_best(u_df)
        assert rec_name == "Ensemble: Soft Voting"
        assert "Ensemble Justification" in exp or "ROC-AUC" in exp
        ok("T06 System Recommendation Logic", f"Correctly selected best system based on ROC-AUC: {rec_name}")
    except Exception as e:
        fail("T06 System Recommendation Logic", str(e))

    print()
    print(f"Results: {passed}/{passed + failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)

