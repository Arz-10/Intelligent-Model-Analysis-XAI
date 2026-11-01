"""
scratch/evaluate_actual_ensembles.py
------------------------------------
Executes full Phase 5 ensemble training and evaluation on the actual DataCo Smart Supply Chain dataset.
"""

import sys, os, time
sys.path.insert(0, r'c:\Users\Aarushi Sontakke\Downloads\CIL_Project')

from src.preprocessing import load_preprocessing_artifacts
from src.model_trainer import load_all_trained_models
from src.evaluator import load_evaluation_results
from src.ensemble import (
    build_and_evaluate_all_ensembles,
    combine_individual_and_ensemble_results,
    recommend_overall_best,
)


def main():
    print("Loading DataCo preprocessed dataset and trained models...")
    arts = load_preprocessing_artifacts()
    models = load_all_trained_models()
    ind_df, _ = load_evaluation_results()

    if arts is None or models is None:
        print("ERROR: Required artifacts missing.")
        return

    X_train = arts.X_train
    y_train = arts.y_train
    X_test  = arts.X_test
    y_test  = arts.y_test

    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    def progress(msg):
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

    print("\nBuilding and evaluating all 4 ensemble strategies...")
    t0 = time.time()
    ens_df, full_metrics, roc_curves = build_and_evaluate_all_ensembles(
        base_models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        progress_callback=progress,
    )

    print(f"\nAll ensembles evaluated in {time.time() - t0:.1f}s.")

    print("\n=== Ensemble Results on DataCo Test Set ===")
    print(ens_df.to_string(index=False))

    unified_df = combine_individual_and_ensemble_results(ind_df, ens_df)
    print("\n=== Unified Model & Ensemble Ranking (by ROC-AUC) ===")
    ranked = unified_df.sort_values(by=["ROC-AUC", "F1-Score"], ascending=[False, False])
    print(ranked[["Model", "Type", "Accuracy", "F1-Score", "ROC-AUC", "Train Time (s)"]].to_string(index=False))

    best_name, explanation = recommend_overall_best(unified_df)
    print(f"\nRecommended System: {best_name}")
    print(explanation)


if __name__ == "__main__":
    main()

