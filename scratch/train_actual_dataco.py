"""
scratch/train_actual_dataco.py
------------------------------
Runs full Phase 4 model training and evaluation on the actual DataCo preprocessed dataset.
"""

import sys, os, time
sys.path.insert(0, r'c:\Users\Aarushi Sontakke\Downloads\CIL_Project')

from src.preprocessing import load_preprocessing_artifacts
from src.model_trainer import train_all_models
from src.evaluator import evaluate_all_models, recommend_best_model, get_roc_curve_data


def main():
    print("Loading preprocessed DataCo dataset...")
    artifacts = load_preprocessing_artifacts()
    if artifacts is None:
        print("ERROR: Preprocessed artifacts not found. Please run preprocessing first.")
        return

    X_train = artifacts.X_train
    y_train = artifacts.y_train
    X_test  = artifacts.X_test
    y_test  = artifacts.y_test

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Features:      {X_train.shape[1]}")

    print("\nStarting training of all 5 classifiers...")
    t_start = time.time()

    def progress(msg):
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

    records = train_all_models(X_train, y_train, progress_callback=progress)

    total_train_time = time.time() - t_start
    print(f"\nAll models finished training in {total_train_time:.1f}s.")

    print("\nEvaluating on held-out test set (54,156 samples)...")
    results_df, full_results = evaluate_all_models(records, X_test, y_test)

    print("\n=== DataCo Test Set Model Comparison ===")
    print(results_df.to_string(index=False))

    print("\nComputing Best Individual Model Recommendation...")
    best_name, explanation = recommend_best_model(results_df)
    print(f"\nRecommended: {best_name}")
    print(explanation)


if __name__ == "__main__":
    main()

