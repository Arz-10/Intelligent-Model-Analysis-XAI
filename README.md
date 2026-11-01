# Intelligent Model Comparison, Ensemble Learning and Explainable AI System

> **Computational Intelligence Lab (CIL) Project**  
> End-to-End Machine Learning, Ensembling, and Explainable AI Pipeline for Supply Chain Late Delivery Risk Prediction.

---

## Project Overview

This system provides a modular, production-grade machine learning and interpretability pipeline for supply chain management using the DataCo Smart Supply Chain dataset:

1. **Dataset Loading and Schema Validation (Phase 2):** Ingestion of 180,519 transactions with 53 raw attributes, schema checking, and target verification (`Late_delivery_risk`).
2. **Preprocessing and Data Leakage Prevention (Phase 3):** Strict exclusion of post-event delivery variables (`Delivery Status`, `Days for shipping (real)`), median imputation, StandardScaler for numerical features, and OrdinalEncoder for categorical features fitted strictly on training data (70/30 stratified split).
3. **Machine Learning Classifiers and Evaluation (Phase 4):** Independent training of 5 classifiers (Logistic Regression, Decision Tree, Random Forest, Calibrated Linear Support Vector Machine, and XGBoost) evaluated on 54,156 held-out test transactions across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. **Ensemble Learning and Out-of-Fold Stacking (Phase 5):** Construction and evaluation of Soft Voting, Weighted Soft Voting (0.70 Random Forest + 0.30 XGBoost), and Stacking with 3-fold out-of-fold cross-validated meta-models.
5. **Explainable AI with SHAP and LIME (Phase 6):** Cooperative game-theoretic SHAP feature importance, impact distribution spreads, local instance waterfall decompositions, and local surrogate LIME decision boundary rules.
6. **Interactive Dashboard:** Complete, emoji-free Streamlit web application with persistent ROC curves and real-time Prediction Explorer.

---

## Project Architecture & Directory Layout

```text
CIL_Project/
├── app.py                     # Main Streamlit web application and page router
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation and setup guide
├── PROJECT_COMPLETION_REPORT.md # Comprehensive final project report
├── .gitignore                 # Version control exclusions (.venv, models, caches)
│
├── data/                      # Dataset repository
│   └── DataCoSupplyChainDataset.csv # 180,519 rows x 53 raw columns
│
├── models/                    # Saved trained model binaries (joblib)
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── svm_calibrated.pkl
│   ├── xgboost.pkl
│   ├── ensemble_soft_voting_top2.pkl
│   ├── ensemble_soft_voting_top3.pkl
│   ├── ensemble_weighted_voting.pkl
│   └── ensemble_stacking.pkl
│
├── outputs/                   # Serialized datasets, metrics, and XAI cache
│   ├── X_train.npy            # 126,363 rows x 36 features
│   ├── X_test.npy             # 54,156 rows x 36 features
│   ├── y_train.npy            # 126,363 labels
│   ├── y_test.npy             # 54,156 labels
│   ├── preprocessing_metadata.json
│   ├── preprocessing_pipeline.pkl
│   ├── model_comparison.csv   # Phase 4 evaluation results
│   ├── model_metrics.json     # Detailed classification reports & confusion matrices
│   ├── roc_curves.json        # Persistent 300-point downsampled ROC coordinates
│   ├── ensemble_comparison.csv# Phase 5 ensemble evaluation results
│   ├── ensemble_metrics.json  # Ensemble metrics and confusion matrices
│   ├── ensemble_roc_curves.json
│   └── shap_global_importance.json # Phase 6 cached global SHAP ranking & beeswarm data
│
├── src/                       # Core modular Python packages
│   ├── data_loader.py         # CSV loading, validation, and missingness reporting
│   ├── preprocessing.py       # Leakage removal, encoding, scaling, stratified splitting
│   ├── model_trainer.py       # Independent model configuration and training
│   ├── evaluator.py           # Evaluation metrics, confusion matrices, ROC persistence
│   ├── ensemble.py            # Soft voting, weighted voting, stacking ensembles
│   └── explainability.py      # SHAP TreeExplainer, LIME surrogate, and domain translation
│
└── scratch/                   # Verification test scripts and utilities
    ├── verify_all_phases.py   # Full end-to-end system test suite (Phases 1-6)
    ├── test_phase3.py         # Unit tests for preprocessing & leakage
    ├── test_phase4.py         # Unit tests for ML classifiers & evaluation
    ├── test_phase5.py         # Unit tests for ensemble methods & ROC persistence
    ├── test_phase6.py         # Unit tests for SHAP, LIME, and Prediction Explorer
    └── build_shap_cache.py    # Utility to precompute global SHAP importance
```

---

## Setup and Installation

### Prerequisites
- Python 3.10 or 3.11
- Windows, macOS, or Linux
- 8 GB RAM or higher recommended

### Step 1: Clone or Navigate to Project
```bash
cd c:\Users\Aarushi Sontakke\Downloads\CIL_Project
```

### Step 2: Set up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify All Modules End-to-End
Run the non-destructive verification suite to ensure all phases are operating cleanly:
```powershell
python scratch/verify_all_phases.py
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Experimental Results & Benchmark Summary

All models were evaluated on the held-out DataCo test set (54,156 transactions) following a 70/30 stratified split. Target variable is `Late_delivery_risk` (1 = Late Delivery, 0 = On Time).

### Individual Classifiers (Phase 4)

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC | Train Time (s) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7236 | **0.8866** | 0.5687 | 0.6929 | **0.8514** | 16.96s | 25 ms |
| **XGBoost** | **0.7311** | 0.8742 | 0.5953 | **0.7083** | 0.8360 | 2.64s | 4 ms |
| **Decision Tree** | 0.7282 | 0.8627 | **0.5997** | 0.7075 | 0.7989 | 2.19s | 3 ms |
| **Support Vector Machine** | 0.6953 | 0.8039 | 0.5876 | 0.6789 | 0.7263 | 18.65s | 8 ms |
| **Logistic Regression** | 0.6895 | 0.7996 | 0.5789 | 0.6716 | 0.7221 | 5.92s | 2 ms |

### Ensemble Models (Phase 5)

| Ensemble Method | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score | ROC-AUC | ROC-AUC Gain over Best Individual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weighted Soft Voting (0.70 RF + 0.30 XGB)** | 0.7243 | **0.8868** | 0.5700 | 0.6939 | **0.8538** | **+0.0024** |
| **Soft Voting (RF + XGBoost)** | 0.7262 | 0.8858 | 0.5746 | 0.6971 | 0.8494 | -0.0020 |
| **Stacking (RF + XGB + LR -> Meta LR)** | **0.7280** | 0.8847 | **0.5795** | **0.7003** | 0.8469 | -0.0045 |
| **Soft Voting (RF + XGBoost + LR)** | 0.7159 | 0.8455 | 0.5896 | 0.6947 | 0.8297 | -0.0217 |

**System Recommendation:**
- **Best Individual Model:** **Random Forest** (ROC-AUC: 0.8514, Precision: 88.66%). Best when operational simplicity, single-model deployment, and false-alarm minimization are required.
- **Best Overall System:** **Weighted Soft Voting Ensemble (0.70 RF + 0.30 XGB)** (ROC-AUC: 0.8538, Precision: 88.68%). Provides maximum discriminatory power and probability calibration across all evaluated configurations.

---

## Explainable AI (XAI) Capabilities (Phase 6)

1. **Global Feature Importance (SHAP):**
   - Ranks all 36 features by Mean Absolute Shapley Value.
   - Identifies that `Shipping Mode` (34.33% relative importance) and `Days for shipment (scheduled)` (31.20% relative importance) represent over 65% of the model's total predictive influence.
   - Precomputed disk caching (`outputs/shap_global_importance.json`) ensures sub-second dashboard rendering.

2. **Feature Impact Distribution Spread:**
   - Scatter-distribution visualization showing how high vs. low feature values drive late delivery risk shifts.
   - Demonstrates that tight scheduled delivery windows (1-2 days) consistently push SHAP values strongly positive (increasing late delivery risk).

3. **Individual Prediction Explorer (Local SHAP):**
   - Allows operators to inspect any test order by index or preset button.
   - Computes base expectation $E[f(X)]$, predicted probability, and positive vs. negative risk drivers.
   - Delivers automated plain-language case explanations for dispatchers.

4. **Local LIME Decision Boundary Rules:**
   - Fits an interpretable local linear surrogate model around the selected transaction.
   - Outputs plain-language threshold rules (e.g., `Condition [Days for shipment (scheduled) <= -0.68] increases late delivery risk by ~19.7%`).
   - Cross-method validation demonstrates high consistency between SHAP and LIME findings.

5. **Ensemble Decomposition:**
   - Explains the Weighted Soft Voting ensemble via linear Shapley decomposition: $\phi_{ensemble} = 0.70 \phi_{RF} + 0.30 \phi_{XGB}$.

---

## System Limitations and Operational Guidelines

1. **Attribution vs. Causation:**
   SHAP and LIME explanations describe mathematical associations captured by the model from historical data. They do not prove physical real-world causation (e.g., courier strikes or extreme weather anomalies not captured in tabular features).
2. **Leakage Prevention Boundaries:**
   Variables recorded after delivery completion (`Delivery Status` and `Days for shipping (real)`) were strictly excluded. If future data pipelines introduce post-event variables, model retraining without those columns is mandatory.
3. **Hardware Considerations:**
   Exact TreeExplainer calculation across tens of thousands of rows is computationally intensive. The system utilizes representative sampling (150-250 instances) for global ranking and precomputed JSON caching to guarantee instant web UI responsiveness.

---

## Project Status

| Phase | Description | Status | Verification Result |
| :---: | :--- | :---: | :---: |
| 1 | Project foundation & environment setup | Complete | Verified [PASS] |
| 2 | Dataset loading & schema validation | Complete | Verified [PASS] (180,519 rows) |
| 3 | Preprocessing, leakage prevention & splitting | Complete | Verified [PASS] (126k / 54k split) |
| 4 | Individual ML model training & evaluation | Complete | Verified [PASS] (5 models trained) |
| 5 | Ensemble learning & persistent ROC curves | Complete | Verified [PASS] (4 ensembles) |
| 6 | Explainability (SHAP, LIME, Prediction Explorer) | Complete | Verified [PASS] (Full XAI suite) |
| 7 | System integration & end-to-end verification | Complete | Verified [PASS] (0 failures) |

---

## License

Developed for the Computational Intelligence Lab (CIL) course. Distributed under standard educational guidelines.
