"""
scratch/restore_dataco_artifacts.py
-----------------------------------
Restores the full, authentic DataCo Smart Supply Chain artifacts across all phases:
1. Data loading and validation (Phase 2)
2. Preprocessing, leakage prevention, train/test split (Phase 3)
3. 5 ML models training and evaluation (Phase 4)
4. 4 Ensemble models training and evaluation (Phase 5)
5. Global SHAP cache generation (Phase 6)
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_dataset
from src.preprocessing import run_preprocessing
from src.model_trainer import train_all_models
from src.evaluator import evaluate_all_models, recommend_best_model
from src.ensemble import build_and_evaluate_all_ensembles
from scratch.build_shap_cache import main as run_build_shap_cache


def main():
    csv_path = os.path.join("data", "DataCoSupplyChainDataset.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at {csv_path}")
        return

    print("=" * 60)
    print("STEP 1: Loading DataCo Smart Supply Chain Dataset...")
    print("=" * 60)
    t0 = time.time()
    df, report = load_dataset(csv_path, source_name="DataCoSupplyChainDataset.csv")
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns in {time.time() - t0:.2f}s.")
    print(f"Target found: {report.get('target_found')} ({report.get('target_column')})")

    print("\n" + "=" * 60)
    print("STEP 2: Running Phase 3 Preprocessing & Leakage Prevention...")
    print("=" * 60)
    t0 = time.time()
    prep_res = run_preprocessing(df)
    print(f"Preprocessing completed in {time.time() - t0:.2f}s.")
    print(f"X_train: {prep_res.X_train.shape}, X_test: {prep_res.X_test.shape}")
    print(f"Features: {len(prep_res.feature_names)}")

    print("\n" + "=" * 60)
    print("STEP 3: Training & Evaluating Phase 4 ML Classifiers...")
    print("=" * 60)
    t0 = time.time()
    records = train_all_models(prep_res.X_train, prep_res.y_train)
    results_df, full_results = evaluate_all_models(records, prep_res.X_test, prep_res.y_test)
    print(f"Model training and evaluation finished in {time.time() - t0:.2f}s.")
    print("\nIndividual Model Performance:")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 4: Training & Evaluating Phase 5 Ensemble Models...")
    print("=" * 60)
    t0 = time.time()
    ens_df, ens_full = build_and_evaluate_all_ensembles(
        prep_res.X_train, prep_res.y_train, prep_res.X_test, prep_res.y_test
    )
    print(f"Ensemble training and evaluation finished in {time.time() - t0:.2f}s.")
    print("\nEnsemble Model Performance:")
    print(ens_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 5: Generating Phase 6 Global SHAP Cache...")
    print("=" * 60)
    run_build_shap_cache()

    print("\n" + "=" * 60)
    print("ALL DATACO ARTIFACTS SUCCESSFULLY RESTORED AND PERSISTED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

