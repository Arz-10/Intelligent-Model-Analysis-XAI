"""
src/evaluator.py
----------------
PURPOSE:
    Computes all evaluation metrics for trained classifiers using the
    held-out test set produced by Phase 3.  Keeps evaluation logic
    entirely separate from training and the UI.

PHASE: 4 — Model Evaluation and Comparison
STATUS: Implemented.

METRICS COMPUTED PER MODEL:
    - Accuracy
    - Precision   (binary, positive_class=1)
    - Recall      (binary, positive_class=1)
    - F1-score    (binary, positive_class=1)
    - ROC-AUC     (using predict_proba if available, else decision_function)
    - Confusion matrix
    - Full classification report (as dict)
    - Training time (seconds, from TrainingRecord)
    - Prediction time (seconds, measured here)

RANKING LOGIC:
    Primary metric  : ROC-AUC  (when valid for all compared models)
    Secondary metric: F1-score  (used as tiebreaker or when ROC-AUC is N/A)

    A model is only recommended if its metrics are derived from the
    actual held-out test set.  No fabricated results.

PUBLIC API:
    evaluate_model(model, X_test, y_test, model_name, train_time_s)
        -> dict
    evaluate_all_models(trained_records, X_test, y_test)
        -> pd.DataFrame, dict[str, dict]
    recommend_best_model(results_df)
        -> str, str  (name, explanation)
    load_evaluation_results()
        -> pd.DataFrame | None
    get_status()
        -> dict
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

OUTPUTS_DIR  = "outputs"
POSITIVE_CLS = 1        # class 1 is "late delivery" — the positive class


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    train_time_s: float = 0.0,
) -> dict:
    """
    Evaluate one fitted model on the test set.

    Parameters
    ----------
    model        : fitted sklearn-compatible estimator
    X_test       : test feature matrix (DataFrame or array)
    y_test       : true labels (Series or array)
    model_name   : display name
    train_time_s : training duration in seconds (from TrainingRecord)

    Returns
    -------
    dict with keys:
        model_name, accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (list[list[int]]),
        classification_report (dict),
        train_time_s, predict_time_s, roc_auc_available (bool),
        error (str or None)
    """
    result = {
        "model_name":            model_name,
        "accuracy":              None,
        "precision":             None,
        "recall":                None,
        "f1":                    None,
        "roc_auc":               None,
        "roc_auc_available":     False,
        "confusion_matrix":      None,
        "classification_report": None,
        "train_time_s":          round(train_time_s, 3),
        "predict_time_s":        None,
        "error":                 None,
    }

    try:
        y_arr = np.asarray(y_test)

        # --- Predictions ---
        t0      = time.perf_counter()
        y_pred  = model.predict(X_test)
        pred_t  = time.perf_counter() - t0
        result["predict_time_s"] = round(pred_t, 4)

        # --- Core metrics ---
        result["accuracy"]  = round(float(accuracy_score(y_arr, y_pred)), 4)
        result["precision"] = round(float(precision_score(
            y_arr, y_pred, pos_label=POSITIVE_CLS, zero_division=0)), 4)
        result["recall"]    = round(float(recall_score(
            y_arr, y_pred, pos_label=POSITIVE_CLS, zero_division=0)), 4)
        result["f1"]        = round(float(f1_score(
            y_arr, y_pred, pos_label=POSITIVE_CLS, zero_division=0)), 4)

        # --- ROC-AUC: try predict_proba first, then decision_function ---
        roc_score = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "predict_proba"):
                try:
                    proba    = model.predict_proba(X_test)[:, 1]
                    roc_score = float(roc_auc_score(y_arr, proba))
                except Exception:
                    pass
            if roc_score is None and hasattr(model, "decision_function"):
                try:
                    scores    = model.decision_function(X_test)
                    roc_score = float(roc_auc_score(y_arr, scores))
                except Exception:
                    pass

        if roc_score is not None:
            result["roc_auc"]           = round(roc_score, 4)
            result["roc_auc_available"] = True

        # --- Confusion matrix ---
        cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
        result["confusion_matrix"] = cm.tolist()

        # --- Classification report ---
        report_dict = classification_report(
            y_arr, y_pred, output_dict=True, zero_division=0
        )
        result["classification_report"] = report_dict

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Evaluate all trained models
# ---------------------------------------------------------------------------

def evaluate_all_models(
    trained_records: dict,
    X_test,
    y_test,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate every successfully trained model and return a comparison table.

    Parameters
    ----------
    trained_records : dict mapping model_name -> TrainingRecord
    X_test, y_test  : held-out test data from Phase 3

    Returns
    -------
    results_df  : pd.DataFrame — one row per model
    full_results: dict         — full metric dict per model (includes confusion matrix)
    """
    full_results: dict = {}
    rows: list         = []

    for name, record in trained_records.items():
        if not record.success or record.model is None:
            # Record the failure but do not compute metrics
            full_results[name] = {
                "model_name":        name,
                "accuracy":          None,
                "precision":         None,
                "recall":            None,
                "f1":                None,
                "roc_auc":           None,
                "roc_auc_available": False,
                "confusion_matrix":  None,
                "train_time_s":      record.train_time_s,
                "predict_time_s":    None,
                "error":             record.error_message or "Training failed.",
            }
            rows.append({
                "Model":           name,
                "Accuracy":        None,
                "Precision":       None,
                "Recall":          None,
                "F1-Score":        None,
                "ROC-AUC":         None,
                "Train Time (s)":  record.train_time_s,
                "Predict Time (s)":None,
                "Status":          "Failed",
            })
            continue

        metrics = evaluate_model(
            model        = record.model,
            X_test       = X_test,
            y_test       = y_test,
            model_name   = name,
            train_time_s = record.train_time_s,
        )
        full_results[name] = metrics

        status = "Completed" if metrics["error"] is None else f"Error: {metrics['error']}"
        rows.append({
            "Model":            name,
            "Accuracy":         metrics["accuracy"],
            "Precision":        metrics["precision"],
            "Recall":           metrics["recall"],
            "F1-Score":         metrics["f1"],
            "ROC-AUC":          metrics["roc_auc"],
            "Train Time (s)":   metrics["train_time_s"],
            "Predict Time (s)": metrics["predict_time_s"],
            "Status":           status,
        })

    results_df = pd.DataFrame(rows)
    _save_evaluation_results(results_df, full_results)

    # Automatically compute and persist ROC curves
    try:
        roc_data = get_roc_curve_data(trained_records, X_test, y_test)
        if roc_data:
            save_roc_curves(roc_data, metadata={"dataset": "DataCo"})
    except Exception:
        pass

    return results_df, full_results


# ---------------------------------------------------------------------------
# Best-model recommendation
# ---------------------------------------------------------------------------

def recommend_best_model(
    results_df: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify the best individual model using a transparent ranking rule.

    Ranking logic
    -------------
    1. Only consider models that completed evaluation without error.
    2. Primary metric:   ROC-AUC (if all successful models have a valid score).
    3. Fallback metric:  F1-Score (if any model lacks a ROC-AUC value).
    4. Secondary metric: F1-Score (tiebreaker when ROC-AUC values are equal).

    Parameters
    ----------
    results_df : pd.DataFrame — output of evaluate_all_models()

    Returns
    -------
    (model_name, explanation_text)
    """
    # Filter to rows with a valid status (not "Failed")
    ok = results_df[results_df["Status"] == "Completed"].copy()

    if ok.empty:
        return "None", "No model completed training and evaluation successfully."

    # Decide primary metric
    all_have_roc = ok["ROC-AUC"].notna().all()
    primary      = "ROC-AUC" if all_have_roc else "F1-Score"

    # Sort: primary metric descending, then F1 descending as tiebreaker
    ok_sorted = ok.sort_values(
        by   = [primary, "F1-Score"],
        ascending = [False, False],
    ).reset_index(drop=True)

    best      = ok_sorted.iloc[0]
    best_name = best["Model"]

    # Build explanation
    lines = [
        f"Recommended model: **{best_name}**",
        "",
        f"Ranking metric: **{primary}**  "
        + ("(used because all models produced valid probability scores)"
           if primary == "ROC-AUC"
           else "(used as fallback because one or more models lacked ROC-AUC scores)"),
        "",
        "Performance of the recommended model on the held-out test set:",
        f"- Accuracy:  {best['Accuracy']:.4f}",
        f"- Precision: {best['Precision']:.4f}",
        f"- Recall:    {best['Recall']:.4f}",
        f"- F1-Score:  {best['F1-Score']:.4f}",
    ]
    if best["ROC-AUC"] is not None:
        lines.append(f"- ROC-AUC:   {best['ROC-AUC']:.4f}")
    lines += [
        "",
        "This is the best individual model measured in this experiment. "
        "Results may differ on other datasets or with different hyperparameters.",
    ]

    # Add trade-off notes
    if best["Recall"] is not None and best["Precision"] is not None:
        if best["Recall"] > best["Precision"] + 0.05:
            lines.append(
                "Note: Recall is notably higher than Precision, meaning this model "
                "tends to flag more late deliveries but may produce more false alarms."
            )
        elif best["Precision"] > best["Recall"] + 0.05:
            lines.append(
                "Note: Precision is notably higher than Recall, meaning this model "
                "is conservative — it misses some late deliveries but rarely produces false alarms."
            )

    return best_name, "\n".join(lines)


# ---------------------------------------------------------------------------
# Probability scores for ROC curves (used by the Streamlit page)
# ---------------------------------------------------------------------------

def get_roc_curve_data(
    trained_records: dict,
    X_test,
    y_test,
) -> dict:
    """
    Compute ROC curve (fpr, tpr, auc) for each model that supports scoring.

    Returns
    -------
    dict mapping model_name -> {"fpr": list, "tpr": list, "auc": float}
    Only models with valid scores are included.
    """
    from sklearn.metrics import roc_curve

    y_arr   = np.asarray(y_test)
    curves  = {}

    for name, record in trained_records.items():
        if not record.success or record.model is None:
            continue
        model = record.model
        scores = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "predict_proba"):
                try:
                    scores = model.predict_proba(X_test)[:, 1]
                except Exception:
                    pass
            if scores is None and hasattr(model, "decision_function"):
                try:
                    scores = model.decision_function(X_test)
                except Exception:
                    pass
        if scores is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_arr, scores, pos_label=POSITIVE_CLS)
            auc         = float(roc_auc_score(y_arr, scores))
            curves[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(auc, 4),
            }
        except Exception:
            pass

    return curves


def save_roc_curves(roc_data: dict, metadata: dict = None, filepath: str = None) -> str:
    """
    Save compact ROC curve coordinates and metadata to outputs/roc_curves.json.

    Parameters
    ----------
    roc_data : dict
        Mapping model_name -> {"fpr": list, "tpr": list, "auc": float}
    metadata : dict, optional
        Extra metadata to record.
    filepath : str, optional
        Target file path (defaults to outputs/roc_curves.json).

    Returns
    -------
    str : full path to saved file
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = filepath or os.path.join(OUTPUTS_DIR, "roc_curves.json")

    compact_data = {}
    for name, curve in roc_data.items():
        fpr = curve.get("fpr", [])
        tpr = curve.get("tpr", [])
        auc = curve.get("auc", 0.0)

        # Downsample if length > 400 for compact JSON and fast UI rendering
        if len(fpr) > 400:
            indices = np.linspace(0, len(fpr) - 1, 300, dtype=int)
            indices = np.unique(indices)
            fpr_compact = [round(float(fpr[i]), 4) for i in indices]
            tpr_compact = [round(float(tpr[i]), 4) for i in indices]
        else:
            fpr_compact = [round(float(x), 4) for x in fpr]
            tpr_compact = [round(float(y), 4) for y in tpr]

        compact_data[name] = {
            "model_name": name,
            "auc": round(float(auc), 4),
            "fpr": fpr_compact,
            "tpr": tpr_compact,
            "num_points": len(fpr_compact),
        }

    payload = {
        "positive_class": POSITIVE_CLS,
        "n_models": len(compact_data),
        "metadata": metadata or {},
        "curves": compact_data,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return out_path


def load_roc_curves(filepath: str = None) -> tuple[dict | None, dict | None]:
    """
    Load saved ROC curves from outputs/roc_curves.json.

    Returns
    -------
    tuple[dict | None, dict | None] : (curves_dict, metadata_dict)
    """
    path = filepath or os.path.join(OUTPUTS_DIR, "roc_curves.json")
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        curves = payload.get("curves", {})
        meta   = payload.get("metadata", {})
        return curves, meta
    except Exception:
        return None, None


def recover_roc_curves() -> tuple[dict | None, str]:
    """
    Controlled evaluation-only recovery path for ROC data:
    1. Returns existing outputs/roc_curves.json if valid.
    2. If missing, loads saved models and test set, computes ROC data,
       persists to outputs/roc_curves.json, and returns it.
    3. If required artifacts are missing, returns (None, explanation).

    Returns
    -------
    tuple[dict | None, str] : (roc_data, status_message)
    """
    curves, _ = load_roc_curves()
    if curves:
        return curves, "Loaded from outputs/roc_curves.json."

    from src.preprocessing import load_preprocessing_artifacts
    from src.model_trainer import load_all_trained_models, TrainingRecord

    arts = load_preprocessing_artifacts()
    if arts is None:
        return None, "Preprocessing artifacts (X_test, y_test) are missing."

    models = load_all_trained_models()
    if not models:
        return None, "Saved model artifacts are missing in models/ directory."

    records = {
        name: TrainingRecord(model_name=name, success=True, model=m)
        for name, m in models.items()
    }

    x_test = arts.X_test if hasattr(arts, "X_test") else arts["X_test"]
    y_test = arts.y_test if hasattr(arts, "y_test") else arts["y_test"]
    roc_data = get_roc_curve_data(records, x_test, y_test)
    if not roc_data:
        return None, "Models did not yield valid scoring outputs."

    save_roc_curves(roc_data, metadata={"recovery_source": "saved_artifacts"})
    return roc_data, "Recovered from saved models and test set without retraining."


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_evaluation_results(
    results_df: pd.DataFrame,
    full_results: dict,
) -> None:
    """Save comparison CSV and full JSON to outputs/."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # CSV — comparison table (drop rows with all-None metrics)
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)

    # JSON — full metrics including confusion matrices
    json_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    serialisable = {}
    for name, m in full_results.items():
        entry = {}
        for k, v in m.items():
            if v is None:
                entry[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                entry[k] = float(v)
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            else:
                entry[k] = v
        serialisable[name] = entry

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)


def load_evaluation_results() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Load saved evaluation results from outputs/.

    Returns (results_df, full_results) or (None, None) if not found.
    """
    csv_path  = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    json_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")

    if not os.path.isfile(csv_path) or not os.path.isfile(json_path):
        return None, None

    results_df   = pd.read_csv(csv_path)
    with open(json_path, encoding="utf-8") as fh:
        full_results = json.load(fh)

    return results_df, full_results


# ---------------------------------------------------------------------------
# Module status
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return readiness status for the Home page phase cards."""
    df, _ = load_evaluation_results()
    if df is None:
        return {
            "module":  "evaluator",
            "phase":   4,
            "ready":   False,
            "message": "No evaluation results found. Train models on the Model Training page.",
        }

    n_ok  = (df["Status"] == "Completed").sum()
    total = len(df)
    return {
        "module":  "evaluator",
        "phase":   4,
        "ready":   n_ok > 0,
        "message": f"{n_ok}/{total} models evaluated successfully.",
    }
