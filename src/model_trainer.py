"""
src/model_trainer.py
--------------------
PURPOSE:
    Trains, saves, and loads the five ML classifiers for Phase 4.
    Reuses the preprocessed X_train / y_train produced by Phase 3.
    Each model is trained independently so that one failure does not
    affect the others.

PHASE: 4 — Machine Learning Model Training
STATUS: Implemented.

MODELS:
    1. Logistic Regression   — sklearn, max_iter=1000, class_weight='balanced'
    2. Decision Tree         — sklearn, max_depth=15,  class_weight='balanced'
    3. Random Forest         — sklearn, n_estimators=200, class_weight='balanced'
    4. Support Vector Machine— sklearn LinearSVC (fast, scales to ~180k rows)
                               Platt-scaling via CalibratedClassifierCV for
                               probability estimates (no test-set leakage —
                               calibration uses cross-validation on train only).
    5. XGBoost               — xgboost.XGBClassifier, scale_pos_weight for imbalance

DESIGN NOTES:
    - All models use RANDOM_STATE = 42 for reproducibility.
    - Class-imbalance is handled in every model (class_weight or scale_pos_weight).
    - SVM: LinearSVC is chosen over kernel SVC because it scales linearly with
      training samples. Kernel SVC with an RBF kernel would be O(n²) memory and
      impractical for 126k rows on a student laptop.
    - XGBoost: imports are guarded; a clear InstallationError is raised if missing.
    - Saved with joblib (no pyarrow dependency).
    - Training metadata is saved to outputs/training_metadata.json.

PUBLIC API:
    build_models(class_ratio)        -> dict[str, estimator]
    train_single_model(name, model, X_train, y_train) -> TrainingRecord
    train_all_models(X_train, y_train) -> dict[str, TrainingRecord]
    save_model(model, filename)      -> str
    load_model(filename)             -> estimator
    load_all_trained_models()        -> dict[str, estimator] | None
    get_status()                     -> dict
"""

import os
import json
import time
import traceback
import warnings
import joblib
import numpy as np

from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR   = "models"
OUTPUTS_DIR  = "outputs"
RANDOM_STATE = 42

MODEL_FILENAMES = {
    "Logistic Regression":    "logistic_regression.pkl",
    "Decision Tree":          "decision_tree.pkl",
    "Random Forest":          "random_forest.pkl",
    "Support Vector Machine": "svm_calibrated.pkl",
    "XGBoost":                "xgboost.pkl",
}


# ---------------------------------------------------------------------------
# Training record
# ---------------------------------------------------------------------------

@dataclass
class TrainingRecord:
    """
    Holds the outcome of training a single model.

    Attributes
    ----------
    model_name    : display name
    success       : True if training completed without error
    model         : fitted estimator, or None on failure
    train_time_s  : wall-clock seconds for .fit()
    artifact_path : path where the model was saved ('' on failure)
    error_message : non-empty string if training failed
    """
    model_name:    str
    success:       bool         = False
    model:         Any          = field(default=None, repr=False)
    train_time_s:  float        = 0.0
    artifact_path: str          = ""
    error_message: str          = ""


# ---------------------------------------------------------------------------
# XGBoost import guard
# ---------------------------------------------------------------------------

def _import_xgboost():
    """
    Import xgboost and return the XGBClassifier class.
    Raises ImportError with a clear install message if missing.
    """
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. "
            "Activate your virtual environment and run:  pip install xgboost"
        )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(class_ratio: float = 1.0) -> dict:
    """
    Instantiate (but do NOT fit) all five classifiers.

    Parameters
    ----------
    class_ratio : float
        Ratio of negative to positive class count in training data
        (negative_count / positive_count).  Used for XGBoost scale_pos_weight.
        A value of 1.0 means balanced classes.

    Returns
    -------
    dict mapping display name -> unfitted estimator
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    # SVM — LinearSVC + Platt calibration for probability estimates.
    # CalibratedClassifierCV uses 5-fold CV on the training set to fit the
    # sigmoid calibrator — no test-set information is used.
    base_svc = LinearSVC(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=0.1,         # smaller C = faster convergence on large datasets
    )
    svm = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")

    # XGBoost
    XGBClassifier = _import_xgboost()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(class_ratio),   # handles class imbalance
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
        n_jobs=-1,
    )

    return {
        "Logistic Regression":    lr,
        "Decision Tree":          dt,
        "Random Forest":          rf,
        "Support Vector Machine": svm,
        "XGBoost":                xgb,
    }


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_single_model(
    name: str,
    model,
    X_train,
    y_train,
    progress_callback=None,
) -> TrainingRecord:
    """
    Fit one model on the training data, save it, and return a TrainingRecord.

    Parameters
    ----------
    name              : str — display name of the model
    model             : unfitted sklearn-compatible estimator
    X_train, y_train  : training data (DataFrames or arrays)
    progress_callback : optional callable(str) — called with status strings

    Returns
    -------
    TrainingRecord
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    record = TrainingRecord(model_name=name)

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    _log(f"Starting: {name}")
    t0 = time.perf_counter()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0

        filename  = MODEL_FILENAMES.get(name, f"{name.lower().replace(' ', '_')}.pkl")
        save_path = save_model(model, filename)

        record.success       = True
        record.model         = model
        record.train_time_s  = round(elapsed, 3)
        record.artifact_path = save_path
        _log(f"Completed: {name}  ({elapsed:.1f}s)")

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record.success       = False
        record.train_time_s  = round(elapsed, 3)
        record.error_message = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED: {name} — {record.error_message}")

    return record


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------

def train_all_models(
    X_train,
    y_train,
    progress_callback=None,
) -> dict[str, TrainingRecord]:
    """
    Train all five classifiers sequentially.

    Parameters
    ----------
    X_train, y_train  : preprocessed training data
    progress_callback : optional callable(str) for status updates

    Returns
    -------
    dict mapping model name -> TrainingRecord
    """
    y_arr = np.asarray(y_train)
    neg   = int((y_arr == 0).sum())
    pos   = int((y_arr == 1).sum())
    class_ratio = neg / pos if pos > 0 else 1.0

    try:
        models = build_models(class_ratio=class_ratio)
    except ImportError as exc:
        # XGBoost missing — build without it
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        if progress_callback:
            progress_callback(f"WARNING: {exc}")

        base_svc = LinearSVC(max_iter=2000, random_state=RANDOM_STATE,
                              class_weight="balanced", C=0.1)
        models = {
            "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                                         class_weight="balanced", n_jobs=-1),
            "Decision Tree":          DecisionTreeClassifier(max_depth=15, min_samples_leaf=10,
                                                             random_state=RANDOM_STATE,
                                                             class_weight="balanced"),
            "Random Forest":          RandomForestClassifier(n_estimators=200, max_depth=20,
                                                             min_samples_leaf=5,
                                                             random_state=RANDOM_STATE,
                                                             class_weight="balanced", n_jobs=-1),
            "Support Vector Machine": CalibratedClassifierCV(base_svc, cv=3, method="sigmoid"),
        }

    records = {}
    for name, model in models.items():
        rec = train_single_model(name, model, X_train, y_train, progress_callback)
        records[name] = rec

    # Save training metadata
    _save_training_metadata(records, X_train, y_train)

    return records


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_model(model, filename: str) -> str:
    """
    Save a fitted model to models/<filename> using joblib.

    Returns the full path of the saved file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path, compress=3)
    return path


def load_model(filename: str):
    """
    Load a model from models/<filename>.

    Raises FileNotFoundError if the file does not exist.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_all_trained_models() -> dict | None:
    """
    Load all five saved models from the models/ directory.

    Returns
    -------
    dict mapping model name -> fitted estimator, or None if any file is missing.
    """
    loaded = {}
    for name, filename in MODEL_FILENAMES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(path):
            return None          # require all models to be present
        try:
            loaded[name] = joblib.load(path)
        except Exception:
            return None
    return loaded


# ---------------------------------------------------------------------------
# Training metadata
# ---------------------------------------------------------------------------

def _save_training_metadata(
    records: dict,
    X_train,
    y_train,
) -> None:
    """Persist a JSON record of training outcomes to outputs/."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    summary = {}
    for name, rec in records.items():
        summary[name] = {
            "success":       rec.success,
            "train_time_s":  rec.train_time_s,
            "artifact_path": rec.artifact_path,
            "error_message": rec.error_message,
        }

    meta = {
        "n_training_rows":  int(len(X_train)),
        "n_features":       int(X_train.shape[1]),
        "feature_names":    list(X_train.columns)
                            if hasattr(X_train, "columns") else [],
        "class_counts":     {
            str(k): int(v)
            for k, v in zip(*np.unique(y_train, return_counts=True))
        },
        "random_state":     RANDOM_STATE,
        "models":           summary,
    }

    out_path = os.path.join(OUTPUTS_DIR, "training_metadata.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Module status
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return readiness status for the Home page phase cards."""
    saved = []
    if os.path.isdir(MODELS_DIR):
        saved = [f for f in os.listdir(MODELS_DIR)
                 if f.endswith(".pkl") and not f.startswith(".")]

    n = len(saved)
    return {
        "module":       "model_trainer",
        "phase":        4,
        "ready":        n == len(MODEL_FILENAMES),
        "saved_models": saved,
        "message": (
            f"All {n} model(s) trained and saved."
            if n == len(MODEL_FILENAMES)
            else (
                f"{n}/{len(MODEL_FILENAMES)} model file(s) found in models/."
                if n > 0
                else "No trained models found. Run training on the Model Training page."
            )
        ),
    }
