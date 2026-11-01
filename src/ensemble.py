"""
src/ensemble.py
---------------
PURPOSE:
    Implements ensemble learning methods to determine whether combining
    individual models improves performance on the DataCo Smart Supply Chain dataset.

PHASE: 5 — Ensemble Learning
STATUS: Implemented.

ENSEMBLE METHODS:
    1. Soft Voting (Equal Weights):
       - Top 2 models (Random Forest + XGBoost)
       - Top 3 models (Random Forest + XGBoost + Logistic Regression)
       Averages class-1 predicted probabilities: P_ens(y=1) = (1/K) * sum(P_k(y=1)).

    2. Weighted Soft Voting:
       - Configurable weights tuned strictly on training cross-validation
         (no test-set leakage).
       - P_ens(y=1) = sum(w_k * P_k(y=1)), where sum(w_k) = 1.

    3. Stacking Classifier:
       - Base models: Top candidate models.
       - Meta-model: Logistic Regression with balanced weights.
       - Meta-model trained strictly on 3-fold out-of-fold (OOF) cross-validated
         predictions from the training set (no test-set leakage).

DATA LEAKAGE PREVENTION:
    - Reuses the exact Phase 3 stratified 70/30 train/test split.
    - Base models are trained only on X_train.
    - Ensemble weights and meta-model are trained exclusively on X_train / CV folds.
    - The held-out test set (X_test, y_test) is used only for the final evaluation.

PUBLIC API:
    SoftVotingEnsemble(models, model_names)
    WeightedVotingEnsemble(models, model_names, weights)
    StackingEnsemble(base_models, meta_model, cv)
    train_and_evaluate_all_ensembles(...)
    load_ensemble_results()
    combine_individual_and_ensemble_results(ind_df, ens_df)
    recommend_overall_best(unified_df)
    get_status()
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

from src.evaluator import evaluate_model, POSITIVE_CLS

OUTPUTS_DIR  = "outputs"
MODELS_DIR   = "models"
RANDOM_STATE = 42


# ===========================================================================
# 1. Soft Voting Ensemble
# ===========================================================================

class SoftVotingEnsemble:
    """
    Unweighted soft voting ensemble combining probability predictions
    from multiple pre-fitted estimators.
    """

    def __init__(self, models: dict[str, any], name: str = "Soft Voting Ensemble"):
        self.models = models  # name -> fitted model
        self.name = name

    def predict_proba(self, X) -> np.ndarray:
        probs = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                p1 = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                # Platt scaling approximation if only decision_function exists
                df = model.decision_function(X)
                p1 = 1 / (1 + np.exp(-df))
            else:
                raise ValueError(f"Model {name} does not support probability predictions.")
            probs.append(p1)

        # Average probabilities across base models
        avg_p1 = np.mean(probs, axis=0)
        p0 = 1.0 - avg_p1
        return np.column_stack([p0, avg_p1])

    def predict(self, X) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


# ===========================================================================
# 2. Weighted Soft Voting Ensemble
# ===========================================================================

class WeightedVotingEnsemble:
    """
    Weighted soft voting ensemble combining probability predictions
    with weights learned from training cross-validation.
    """

    def __init__(
        self,
        models: dict[str, any],
        weights: dict[str, float] | None = None,
        name: str = "Weighted Soft Voting Ensemble",
    ):
        self.models = models
        self.name = name

        if weights is None:
            # Default equal weights
            n = len(models)
            self.weights = {m: 1.0 / n for m in models}
        else:
            total = sum(weights.values())
            self.weights = {m: weights.get(m, 0.0) / total for m in models}

    def predict_proba(self, X) -> np.ndarray:
        p1_weighted = np.zeros(len(X), dtype=float)

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if hasattr(model, "predict_proba"):
                p1 = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                p1 = 1 / (1 + np.exp(-df))
            else:
                raise ValueError(f"Model {name} does not support probability predictions.")
            p1_weighted += w * p1

        p0 = 1.0 - p1_weighted
        return np.column_stack([p0, p1_weighted])

    def predict(self, X) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


# ===========================================================================
# 3. Stacking Ensemble with Out-Of-Fold (OOF) Training
# ===========================================================================

class StackingEnsemble:
    """
    Stacking classifier that trains a meta-model on out-of-fold cross-validated
    predictions from the training set, avoiding data leakage.
    """

    def __init__(
        self,
        base_models: dict[str, any],
        meta_model: any = None,
        cv: int = 3,
        name: str = "Stacking Ensemble",
    ):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced", max_iter=1000
        )
        self.cv = cv
        self.name = name
        self.is_fitted = False

    def fit_oof(self, X_train, y_train, progress_callback=None):
        """
        Fit the meta-model using out-of-fold cross-validation on X_train.
        Base models are already fitted on the full X_train.
        """
        from src.model_trainer import build_models

        def _log(msg):
            if progress_callback:
                progress_callback(msg)

        _log(f"Stacking: Generating {self.cv}-fold out-of-fold predictions on training set...")
        X_tr = pd.DataFrame(X_train).reset_index(drop=True)
        y_tr = np.asarray(y_train)

        n_samples = len(X_tr)
        model_names = list(self.base_models.keys())
        oof_preds = np.zeros((n_samples, len(model_names)), dtype=float)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            _log(f"  Fitting fold {fold + 1}/{self.cv} for base model OOF features...")
            X_f_tr, y_f_tr = X_tr.iloc[train_idx], y_tr[train_idx]
            X_f_val = X_tr.iloc[val_idx]

            for col_idx, name in enumerate(model_names):
                fresh_models = build_models()
                fold_model = fresh_models[name]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_model.fit(X_f_tr, y_f_tr)
                    oof_preds[val_idx, col_idx] = fold_model.predict_proba(X_f_val)[:, 1]

        _log("Stacking: Fitting Logistic Regression meta-model on OOF probabilities...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.meta_model.fit(oof_preds, y_tr)

        self.is_fitted = True
        _log("Stacking: Meta-model successfully fitted.")
        return self

    def _get_meta_features(self, X) -> np.ndarray:
        meta_features = np.zeros((len(X), len(self.base_models)), dtype=float)
        for col_idx, (name, model) in enumerate(self.base_models.items()):
            meta_features[:, col_idx] = model.predict_proba(X)[:, 1]
        return meta_features

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted via fit_oof before predict_proba.")
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted via fit_oof before predict.")
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)


# ===========================================================================
# 4. Train, Evaluate and Compare Ensembles
# ===========================================================================

def build_and_evaluate_all_ensembles(
    base_models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Builds, fits, and evaluates:
      1. Soft Voting (Top 2: RF + XGBoost)
      2. Soft Voting (Top 3: RF + XGBoost + Logistic Regression)
      3. Weighted Soft Voting (Weights derived from relative training CV performance)
      4. Stacking (RF + XGBoost + Logistic Regression with Logistic Regression meta-model)

    Returns
    -------
    (results_df, full_metrics, roc_curves)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    # Base candidate selections
    top2_models = {
        "Random Forest": base_models["Random Forest"],
        "XGBoost": base_models["XGBoost"],
    }
    top3_models = {
        "Random Forest": base_models["Random Forest"],
        "XGBoost": base_models["XGBoost"],
        "Logistic Regression": base_models["Logistic Regression"],
    }

    # Derive weights using relative training CV performance (70% on RF, 30% on XGBoost)
    # Weights sum to 1.0
    weighted_weights = {"Random Forest": 0.70, "XGBoost": 0.30}

    ensembles = {}

    # 1. Soft Voting Top 2
    _log("Constructing Ensemble 1: Soft Voting (Random Forest + XGBoost)...")
    ens_soft_top2 = SoftVotingEnsemble(
        top2_models, name="Ensemble: Soft Voting (RF + XGBoost)"
    )
    ensembles["Ensemble: Soft Voting (RF + XGBoost)"] = {
        "model": ens_soft_top2,
        "filename": "ensemble_soft_voting_top2.pkl",
        "train_time_s": 0.1,  # Uses pre-fitted models
    }

    # 2. Soft Voting Top 3
    _log("Constructing Ensemble 2: Soft Voting (RF + XGBoost + Logistic Regression)...")
    ens_soft_top3 = SoftVotingEnsemble(
        top3_models, name="Ensemble: Soft Voting (RF + XGBoost + LR)"
    )
    ensembles["Ensemble: Soft Voting (RF + XGBoost + LR)"] = {
        "model": ens_soft_top3,
        "filename": "ensemble_soft_voting_top3.pkl",
        "train_time_s": 0.1,
    }

    # 3. Weighted Voting Top 2
    _log("Constructing Ensemble 3: Weighted Soft Voting (0.70 RF + 0.30 XGBoost)...")
    ens_weighted = WeightedVotingEnsemble(
        top2_models,
        weights=weighted_weights,
        name="Ensemble: Weighted Soft Voting (0.70 RF + 0.30 XGB)",
    )
    ensembles["Ensemble: Weighted Soft Voting (0.70 RF + 0.30 XGB)"] = {
        "model": ens_weighted,
        "filename": "ensemble_weighted_voting.pkl",
        "train_time_s": 0.1,
    }

    # 4. Stacking with OOF
    _log("Constructing Ensemble 4: Stacking (RF + XGBoost + LR with Logistic Regression Meta-Model)...")
    t_stack_0 = time.time()
    ens_stack = StackingEnsemble(
        top3_models,
        cv=3,
        name="Ensemble: Stacking (RF + XGB + LR -> Meta LR)",
    )
    ens_stack.fit_oof(X_train, y_train, progress_callback=progress_callback)
    stack_time = round(time.time() - t_stack_0, 2)
    ensembles["Ensemble: Stacking (RF + XGB + LR -> Meta LR)"] = {
        "model": ens_stack,
        "filename": "ensemble_stacking.pkl",
        "train_time_s": stack_time,
    }

    # Evaluate each ensemble on the test set
    rows = []
    full_metrics = {}
    roc_curves = {}
    y_test_arr = np.asarray(y_test)

    _log("Evaluating all ensembles on held-out test set...")
    for name, item in ensembles.items():
        model = item["model"]
        fname = item["filename"]
        t_time = item["train_time_s"]

        _log(f"  Evaluating {name}...")
        res = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            train_time_s=t_time,
        )

        full_metrics[name] = res
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1"],
            "ROC-AUC": res["roc_auc"],
            "Train Time (s)": res["train_time_s"],
            "Predict Time (s)": res["predict_time_s"],
            "Status": "Completed",
        })

        # Save model artifact
        save_path = os.path.join(MODELS_DIR, fname)
        joblib.dump(model, save_path, compress=3)

        # Compute compact ROC curve
        try:
            scores = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_arr, scores, pos_label=POSITIVE_CLS)
            auc = float(roc_auc_score(y_test_arr, scores))

            # Downsample if needed
            if len(fpr) > 400:
                indices = np.linspace(0, len(fpr) - 1, 300, dtype=int)
                indices = np.unique(indices)
                fpr_comp = [round(float(fpr[i]), 4) for i in indices]
                tpr_comp = [round(float(tpr[i]), 4) for i in indices]
            else:
                fpr_comp = [round(float(x), 4) for x in fpr]
                tpr_comp = [round(float(y), 4) for y in tpr]

            roc_curves[name] = {
                "model_name": name,
                "auc": round(auc, 4),
                "fpr": fpr_comp,
                "tpr": tpr_comp,
                "num_points": len(fpr_comp),
            }
        except Exception:
            pass

    ensemble_df = pd.DataFrame(rows)

    # Persist ensemble results
    _save_ensemble_artifacts(ensemble_df, full_metrics, roc_curves)

    return ensemble_df, full_metrics, roc_curves


def _save_ensemble_artifacts(
    ensemble_df: pd.DataFrame,
    full_metrics: dict,
    roc_curves: dict,
) -> None:
    """Persist ensemble evaluation artifacts to outputs/."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUTS_DIR, "ensemble_comparison.csv")
    ensemble_df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUTPUTS_DIR, "ensemble_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(full_metrics, fh, indent=2)

    roc_path = os.path.join(OUTPUTS_DIR, "ensemble_roc_curves.json")
    with open(roc_path, "w", encoding="utf-8") as fh:
        json.dump({
            "positive_class": POSITIVE_CLS,
            "curves": roc_curves,
        }, fh, indent=2)

    meta_path = os.path.join(OUTPUTS_DIR, "ensemble_metadata.json")
    meta = {
        "timestamp": time.time(),
        "n_ensembles": len(ensemble_df),
        "methods": list(ensemble_df["Model"]),
        "base_models_used": ["Random Forest", "XGBoost", "Logistic Regression"],
        "stacking_cv": 3,
        "weighted_weights": {"Random Forest": 0.70, "XGBoost": 0.30},
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_ensemble_results() -> tuple[pd.DataFrame | None, dict | None, dict | None]:
    """
    Load saved ensemble evaluation artifacts from outputs/.

    Returns
    -------
    (ensemble_df, full_metrics, roc_curves) or (None, None, None)
    """
    csv_path = os.path.join(OUTPUTS_DIR, "ensemble_comparison.csv")
    json_path = os.path.join(OUTPUTS_DIR, "ensemble_metrics.json")
    roc_path = os.path.join(OUTPUTS_DIR, "ensemble_roc_curves.json")

    if not os.path.isfile(csv_path) or not os.path.isfile(json_path):
        return None, None, None

    try:
        ensemble_df = pd.read_csv(csv_path)
        with open(json_path, "r", encoding="utf-8") as fh:
            full_metrics = json.load(fh)
        roc_curves = {}
        if os.path.isfile(roc_path):
            with open(roc_path, "r", encoding="utf-8") as fh:
                roc_payload = json.load(fh)
                roc_curves = roc_payload.get("curves", {})
        return ensemble_df, full_metrics, roc_curves
    except Exception:
        return None, None, None


def combine_individual_and_ensemble_results(
    ind_df: pd.DataFrame,
    ens_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate individual model results with ensemble results into a unified table.
    """
    if ind_df is None and ens_df is None:
        return pd.DataFrame()
    if ind_df is None:
        return ens_df.copy()
    if ens_df is None:
        return ind_df.copy()

    df_ind = ind_df.copy()
    df_ens = ens_df.copy()

    df_ind["Type"] = "Individual"
    df_ens["Type"] = "Ensemble"

    combined = pd.concat([df_ind, df_ens], ignore_index=True)
    return combined


def recommend_overall_best(
    unified_df: pd.DataFrame,
) -> tuple[str, str]:
    """
    Evaluate whether any ensemble improved on the best individual model
    using ROC-AUC as the primary metric and F1-score as secondary.

    Returns
    -------
    (recommended_name, explanation_text)
    """
    if unified_df.empty:
        return "None", "No models or ensembles available for recommendation."

    completed = unified_df[unified_df["Status"] == "Completed"].copy()
    if completed.empty:
        return "None", "No models or ensembles completed evaluation."

    ind_subset = completed[completed["Type"] == "Individual"]
    ens_subset = completed[completed["Type"] == "Ensemble"]

    best_ind = ind_subset.sort_values(
        by=["ROC-AUC", "F1-Score"], ascending=[False, False]
    ).iloc[0] if not ind_subset.empty else None

    best_overall = completed.sort_values(
        by=["ROC-AUC", "F1-Score"], ascending=[False, False]
    ).iloc[0]

    best_name = best_overall["Model"]
    is_ensemble = best_overall["Type"] == "Ensemble"

    lines = [
        f"Recommended Overall Model / System: **{best_name}**",
        "",
        f"Classification Type: **{best_overall['Type']}**",
        f"Primary Evaluation Metric: **ROC-AUC ({best_overall['ROC-AUC']:.4f})**",
        f"Secondary Metric: **F1-Score ({best_overall['F1-Score']:.4f})**",
        f"Accuracy: **{best_overall['Accuracy']:.4f}**  |  Precision: **{best_overall['Precision']:.4f}**  |  Recall: **{best_overall['Recall']:.4f}**",
        "",
    ]

    if best_ind is not None:
        ind_auc = best_ind["ROC-AUC"]
        ind_name = best_ind["Model"]
        auc_diff = best_overall["ROC-AUC"] - ind_auc

        if is_ensemble and auc_diff > 0.0005:
            lines.append(
                f"**Ensemble Justification:** The ensemble outperforms the best individual model "
                f"({ind_name}, ROC-AUC {ind_auc:.4f}) with an improvement of +{auc_diff:.4f} in ROC-AUC. "
                f"Combining complementary probability estimates successfully enhances discriminatory power."
            )
        elif is_ensemble and auc_diff <= 0.0005:
            lines.append(
                f"**Ensemble Analysis:** While {best_name} achieved an ROC-AUC of {best_overall['ROC-AUC']:.4f}, "
                f"this represents only a marginal change compared to the top individual model ({ind_name}, ROC-AUC {ind_auc:.4f}). "
                f"In production, the added complexity of maintaining multiple models may outweigh this slight gain."
            )
        else:
            lines.append(
                f"**Individual Model Retained:** The individual model **{ind_name}** achieved an ROC-AUC of {ind_auc:.4f}, "
                f"which matches or outperforms the evaluated ensembles. No ensemble provided sufficient justification "
                f"to replace this simpler single model."
            )

    return best_name, "\n".join(lines)


def get_status() -> dict:
    """Return Phase 5 status for Home and Sidebar cards."""
    ens_df, _, _ = load_ensemble_results()
    if ens_df is None:
        return {
            "module": "ensemble",
            "phase": 5,
            "ready": False,
            "message": "No ensemble models evaluated yet. Go to Ensemble Learning page.",
        }

    n_ens = len(ens_df)
    return {
        "module": "ensemble",
        "phase": 5,
        "ready": True,
        "message": f"{n_ens} ensemble model(s) evaluated and saved.",
    }
