"""
src/explainability.py
---------------------
PURPOSE:
    Provides model interpretability and explainability using SHAP (Shapley
    Additive exPlanations) and LIME (Local Interpretable Model-agnostic
    Explanations) for the DataCo Smart Supply Chain classification models.

PHASE: 6 — Explainability & Prediction Explorer
STATUS: Fully Implemented.

EXPLAINABILITY CAPABILITIES:
    1. Global Feature Importance (SHAP):
       - TreeExplainer on held-out test data samples
       - Mean absolute SHAP values ranked across all features
       - Feature impact direction (positive / negative correlation with risk)
       - Summary beeswarm distribution data for interactive visualization
       - Precomputed disk caching (outputs/shap_global_importance.json) for
         instant rendering in Streamlit.

    2. Local Instance Explanations (SHAP):
       - Exact Shapley values for individual customer orders
       - Base value E[f(X)], prediction probability, and attribution breakdown
       - Top positive drivers (factors increasing late delivery risk)
       - Top negative drivers (factors mitigating late delivery risk)
       - Support for tree models (Random Forest, XGBoost, Decision Tree).

    3. Local Instance Explanations (LIME):
       - LimeTabularExplainer fitted on training background samples
       - Local linear surrogate approximation around individual test instances
       - Threshold condition rules (e.g., 'Days for shipment <= 2')
       - Plain-language interpretations of local feature rules.

    4. Ensemble Explanation (Weighted Soft Voting):
       - Linear decomposition of SHAP values: phi_ensemble = w_RF * phi_RF + w_XGB * phi_XGB
       - Direct black-box LIME explanation using ensemble.predict_proba
       - Component-by-component comparison (Random Forest vs XGBoost).

    5. Supply Chain Domain Interpretability:
       - Technical explanations accompanied by clear business context
       - Distinction between mathematical model attribution and physical causation.

PUBLIC API:
    get_status() -> dict
    get_tree_explainer(model) -> shap.TreeExplainer
    get_lime_explainer(X_train_sample, feature_names) -> LimeTabularExplainer
    compute_instance_shap(model, x_row, feature_names, explainer) -> dict
    compute_instance_lime(explainer, predict_fn, x_row, num_features) -> dict
    compute_instance_ensemble_shap(rf_model, xgb_model, weights, x_row, feature_names) -> dict
    compute_global_shap_importance(model, X_sample, feature_names, model_name) -> dict
    save_global_shap_cache(cache_dict, filepath) -> str
    load_global_shap_cache(filepath) -> dict | None
    get_or_compute_global_shap(...) -> dict
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

# Guarded imports for SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LimeTabularExplainer = None
    LIME_AVAILABLE = False


DEFAULT_CACHE_PATH = os.path.join("outputs", "shap_global_importance.json")

__all__ = [
    "DEFAULT_CACHE_PATH",
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
    "get_status",
    "get_tree_explainer",
    "get_lime_explainer",
    "compute_global_shap_importance",
    "save_global_shap_cache",
    "load_global_shap_cache",
    "get_or_compute_global_shap",
    "compute_instance_shap",
    "compute_instance_lime",
    "compute_instance_ensemble_shap",
    "get_domain_context",
    "DOMAIN_INTERPRETATIONS",
]


# ---------------------------------------------------------------------------
# Status and Availability
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return status of explainability tools and saved cache."""
    cache_exists = os.path.exists(DEFAULT_CACHE_PATH)
    cached_models = []
    if cache_exists:
        try:
            with open(DEFAULT_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                cached_models = list(data.keys())
        except Exception:
            pass

    return {
        "shap_available": SHAP_AVAILABLE,
        "lime_available": LIME_AVAILABLE,
        "cache_exists": cache_exists,
        "cache_path": DEFAULT_CACHE_PATH,
        "cached_models": cached_models,
    }


# ---------------------------------------------------------------------------
# Explainer Factory Functions
# ---------------------------------------------------------------------------

def get_tree_explainer(model):
    """
    Instantiate and return a SHAP TreeExplainer for tree-based models.

    Parameters
    ----------
    model : fitted tree estimator (RandomForestClassifier, XGBClassifier, etc.)

    Returns
    -------
    shap.TreeExplainer
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP library is not installed in the current environment. "
            "Install it via: pip install shap"
        )
    return shap.TreeExplainer(model)


def get_lime_explainer(
    X_train_sample: np.ndarray,
    feature_names: list,
    class_names: list = None,
    random_state: int = 42,
):
    """
    Instantiate and return a LIME LimeTabularExplainer.

    Parameters
    ----------
    X_train_sample : np.ndarray
        Representative background sample from training data (e.g. 100-300 rows).
    feature_names : list[str]
        Ordered feature names matching X_train_sample columns.
    class_names : list[str], optional
        Display names for target classes (default: ['On Time', 'Late']).
    random_state : int
        Seed for reproducible perturbations.

    Returns
    -------
    LimeTabularExplainer
    """
    if not LIME_AVAILABLE:
        raise ImportError(
            "LIME library is not installed in the current environment. "
            "Install it via: pip install lime"
        )

    if class_names is None:
        class_names = ["On Time", "Late"]

    # Convert to plain numpy 2D array if dataframe
    if hasattr(X_train_sample, "values"):
        X_train_sample = X_train_sample.values

    return LimeTabularExplainer(
        training_data=np.asarray(X_train_sample, dtype=float),
        feature_names=list(feature_names),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Helper: Standardize SHAP Values for Binary Classification (Class 1)
# ---------------------------------------------------------------------------

def _extract_class1_shap(shap_raw) -> np.ndarray:
    """
    Standardize raw SHAP values from TreeExplainer to a 2D numpy array
    representing class 1 (Late delivery risk = 1).
    """
    if isinstance(shap_raw, list):
        # List of [shap_class0, shap_class1]
        vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
    elif hasattr(shap_raw, "values"):
        # shap.Explanation object
        v = shap_raw.values
        if v.ndim == 3 and v.shape[2] == 2:
            vals = v[:, :, 1]
        else:
            vals = v
    else:
        vals = np.asarray(shap_raw)
        if vals.ndim == 3 and vals.shape[2] == 2:
            vals = vals[:, :, 1]

    # Ensure 2D array
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


# ---------------------------------------------------------------------------
# Global SHAP Feature Importance
# ---------------------------------------------------------------------------

def compute_global_shap_importance(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str = "random_forest",
    explainer=None,
) -> dict:
    """
    Compute global feature importance across an evaluation sample using SHAP.

    Parameters
    ----------
    model : fitted estimator
    X_sample : np.ndarray
        Sample of evaluation data (e.g. 200-300 rows from X_test).
    feature_names : list[str]
        Ordered feature names.
    model_name : str
        Identifier for the model.
    explainer : shap.TreeExplainer, optional

    Returns
    -------
    dict with:
        - 'model_name': str
        - 'n_samples': int
        - 'ranking': list[dict]
        - 'summary_distribution': list[dict] (for beeswarm / spread plots)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available.")

    if explainer is None:
        explainer = get_tree_explainer(model)

    # Compute raw SHAP values
    X_arr = np.asarray(X_sample, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_sv = explainer.shap_values(X_arr)

    sv_class1 = _extract_class1_shap(raw_sv)  # shape (n_samples, n_features)

    # 1. Mean Absolute SHAP per feature
    mean_abs = np.mean(np.abs(sv_class1), axis=0)
    total_abs = np.sum(mean_abs)
    if total_abs == 0:
        total_abs = 1e-9

    # 2. Determine general direction (correlation between feature value and SHAP value)
    directions = []
    for j in range(len(feature_names)):
        col_vals = X_arr[:, j]
        col_shap = sv_class1[:, j]
        if np.std(col_vals) > 1e-6 and np.std(col_shap) > 1e-6:
            corr = np.corrcoef(col_vals, col_shap)[0, 1]
            if corr > 0.15:
                direction = "Higher values increase late risk"
            elif corr < -0.15:
                direction = "Higher values reduce late risk"
            else:
                direction = "Non-linear / context-dependent"
        else:
            direction = "Constant / uniform impact"
        directions.append(direction)

    # Sort descending by importance
    sorted_indices = np.argsort(mean_abs)[::-1]

    ranking = []
    for rank, idx in enumerate(sorted_indices, start=1):
        rel_pct = (mean_abs[idx] / total_abs) * 100.0
        ranking.append({
            "rank": rank,
            "feature": feature_names[idx],
            "feature_index": int(idx),
            "mean_abs_shap": float(round(mean_abs[idx], 5)),
            "relative_importance_pct": float(round(rel_pct, 2)),
            "impact_direction": directions[idx],
        })

    # 3. Summary distribution data for top 15 features (for interactive beeswarm / box plot)
    top15_indices = sorted_indices[:15]
    summary_distribution = []
    for idx in top15_indices:
        feat_vals = X_arr[:, idx]
        feat_shap = sv_class1[:, idx]
        # Min-max scale feature values to [0, 1] for visual color coding
        val_min, val_max = np.min(feat_vals), np.max(feat_vals)
        if val_max > val_min:
            norm_vals = (feat_vals - val_min) / (val_max - val_min)
        else:
            norm_vals = np.full_like(feat_vals, 0.5)

        summary_distribution.append({
            "feature": feature_names[idx],
            "mean_abs_shap": float(round(mean_abs[idx], 5)),
            "shap_values": [float(round(v, 4)) for v in feat_shap],
            "normalized_values": [float(round(v, 3)) for v in norm_vals],
        })

    return {
        "model_name": model_name,
        "n_samples": int(len(X_arr)),
        "ranking": ranking,
        "summary_distribution": summary_distribution,
    }


def save_global_shap_cache(cache_dict: dict, filepath: str = DEFAULT_CACHE_PATH) -> str:
    """Save global SHAP importance results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, indent=2)
    return filepath


def load_global_shap_cache(filepath: str = DEFAULT_CACHE_PATH) -> dict | None:
    """
    Load precomputed global SHAP results from JSON.

    Parameters
    ----------
    filepath : str
        Path to JSON cache file (defaults to outputs/shap_global_importance.json).

    Returns
    -------
    dict | None
        Cached dictionary containing model rankings and beeswarm distribution,
        or None if file is missing, empty, or unparseable.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data:
                return data
            return None
    except Exception:
        return None


def get_or_compute_global_shap(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str = "random_forest",
    filepath: str = DEFAULT_CACHE_PATH,
    force_recompute: bool = False,
) -> dict:
    """
    Retrieve global SHAP feature importance from cache or compute safely on demand.

    Parameters
    ----------
    model : fitted tree estimator
    X_sample : np.ndarray
        Representative background/evaluation sample.
    feature_names : list[str]
    model_name : str
    filepath : str
    force_recompute : bool

    Returns
    -------
    dict with 'model_name', 'n_samples', 'ranking', 'summary_distribution'.
    """
    if not force_recompute:
        cache = load_global_shap_cache(filepath)
        if cache is not None and model_name in cache:
            return cache[model_name]

    # Compute using existing SHAP pipeline
    importance_result = compute_global_shap_importance(
        model, X_sample, feature_names, model_name=model_name
    )

    # Load existing cache or create fresh dict
    cache = load_global_shap_cache(filepath) or {}
    cache[model_name] = importance_result
    save_global_shap_cache(cache, filepath)

    return importance_result


# ---------------------------------------------------------------------------
# Local Instance SHAP Explanation
# ---------------------------------------------------------------------------

def compute_instance_shap(
    model,
    x_row: np.ndarray,
    feature_names: list,
    explainer=None,
    top_k: int = 5,
) -> dict:
    """
    Compute exact SHAP explanation for a single customer order instance.

    Parameters
    ----------
    model : fitted estimator
    x_row : 1D array of shape (n_features,)
    feature_names : list[str]
    explainer : shap.TreeExplainer, optional
    top_k : int, default 5

    Returns
    -------
    dict with:
        - 'base_value': float (expected value)
        - 'predicted_prob': float (prob of class 1)
        - 'top_positive': list[dict] (factors increasing late risk)
        - 'top_negative': list[dict] (factors decreasing late risk)
        - 'all_contributions': list[dict]
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available.")

    if explainer is None:
        explainer = get_tree_explainer(model)

    x_2d = np.asarray(x_row, dtype=float).reshape(1, -1)

    # Base value
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_val = float(expected[1] if len(expected) > 1 else expected[0])
    else:
        base_val = float(expected)

    # Compute raw SHAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_sv = explainer.shap_values(x_2d)

    sv_class1 = _extract_class1_shap(raw_sv)[0]  # shape (n_features,)

    # Predicted probability
    predicted_prob = None
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Wrap as DataFrame if model has feature_names_in_
            if hasattr(model, "feature_names_in_"):
                df_temp = pd.DataFrame(x_2d, columns=model.feature_names_in_)
                prob_arr = model.predict_proba(df_temp)[0]
            else:
                prob_arr = model.predict_proba(x_2d)[0]
            predicted_prob = float(prob_arr[1])

    # Assemble all contributions
    all_contributions = []
    for i, name in enumerate(feature_names):
        shap_val = float(sv_class1[i])
        raw_val = float(x_2d[0, i])   # use the numpy array already computed above
        all_contributions.append({
            "feature": name,
            "feature_index": i,
            "value": round(raw_val, 4),
            "shap_value": round(shap_val, 5),
            "abs_shap": abs(shap_val),
            "direction": "Increases Late Risk" if shap_val > 0 else "Reduces Late Risk",
        })

    # Sort positive (increasing risk) and negative (reducing risk)
    positive_drivers = sorted(
        [c for c in all_contributions if c["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:top_k]

    negative_drivers = sorted(
        [c for c in all_contributions if c["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:top_k]

    return {
        "base_value": round(base_val, 4),
        "predicted_prob": round(predicted_prob, 4) if predicted_prob is not None else None,
        "top_positive": positive_drivers,
        "top_negative": negative_drivers,
        "all_contributions": sorted(all_contributions, key=lambda x: x["abs_shap"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Local Instance LIME Explanation
# ---------------------------------------------------------------------------

def compute_instance_lime(
    explainer,
    predict_fn,
    x_row: np.ndarray,
    num_features: int = 8,
) -> dict:
    """
    Compute local explanation for a single customer order using LIME.

    Parameters
    ----------
    explainer : LimeTabularExplainer
    predict_fn : callable (accepts 2D numpy array and returns 2D probabilities)
    x_row : 1D array of shape (n_features,)
    num_features : int, default 8

    Returns
    -------
    dict with:
        - 'prediction_prob_late': float
        - 'rules': list[dict] with rule, weight, direction, interpretation
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is not available.")

    x_1d = np.asarray(x_row, dtype=float).ravel()

    # Wrap predict_fn to ensure clean execution without scikit-learn warnings
    def safe_predict(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_obj = getattr(predict_fn, "__self__", None)
            if model_obj is not None and hasattr(model_obj, "feature_names_in_"):
                df_arr = pd.DataFrame(arr, columns=model_obj.feature_names_in_)
                return predict_fn(df_arr)
            return predict_fn(arr)

    # Generate LIME explanation for class 1
    exp = explainer.explain_instance(
        data_row=x_1d,
        predict_fn=safe_predict,
        num_features=num_features,
        labels=(1,),
    )

    # Extract list of (rule_string, weight) for class 1
    raw_list = exp.as_list(label=1)
    rules = []
    for rule, weight in raw_list:
        w = float(weight)
        direction = "Increases Late Risk" if w > 0 else "Reduces Late Risk"
        impact_pct = round(abs(w) * 100.0, 1)

        if w > 0:
            interpretation = f"Condition [{rule}] increases predicted late delivery risk by ~{impact_pct}%."
        else:
            interpretation = f"Condition [{rule}] reduces predicted late delivery risk by ~{impact_pct}%."

        rules.append({
            "rule": rule,
            "weight": round(w, 5),
            "direction": direction,
            "impact_pct": impact_pct,
            "interpretation": interpretation,
        })

    # Sort descending by absolute weight
    rules.sort(key=lambda x: abs(x["weight"]), reverse=True)

    # Current model probability for this instance
    p_late = None
    try:
        p_arr = safe_predict(x_1d.reshape(1, -1))[0]
        p_late = float(round(p_arr[1], 4))
    except Exception:
        pass

    return {
        "prediction_prob_late": p_late,
        "rules": rules,
    }


# ---------------------------------------------------------------------------
# Ensemble Explanation (Weighted Soft Voting)
# ---------------------------------------------------------------------------

def compute_instance_ensemble_shap(
    rf_model,
    xgb_model,
    weights: dict,
    x_row: np.ndarray,
    feature_names: list,
    rf_explainer=None,
    xgb_explainer=None,
    top_k: int = 5,
) -> dict:
    """
    Compute decomposed SHAP explanation for the Weighted Soft Voting Ensemble:
        P_ens = w_RF * P_RF + w_XGB * P_XGB
        phi_ens = w_RF * phi_RF + w_XGB * phi_XGB

    Parameters
    ----------
    rf_model : fitted RandomForestClassifier
    xgb_model : fitted XGBClassifier
    weights : dict, e.g. {'Random Forest': 0.70, 'XGBoost': 0.30}
    x_row : 1D array
    feature_names : list[str]
    rf_explainer, xgb_explainer : optional pre-instantiated explainers
    top_k : int, default 5

    Returns
    -------
    dict with ensemble combined attributions and individual component comparisons.
    """
    w_rf = float(weights.get("Random Forest", 0.70))
    w_xgb = float(weights.get("XGBoost", 0.30))
    x_1d = np.asarray(x_row, dtype=float).ravel()

    rf_exp = compute_instance_shap(rf_model, x_row, feature_names, explainer=rf_explainer, top_k=top_k)
    xgb_exp = compute_instance_shap(xgb_model, x_row, feature_names, explainer=xgb_explainer, top_k=top_k)

    # Combined base value
    comb_base = w_rf * rf_exp["base_value"] + w_xgb * xgb_exp["base_value"]

    # Combined probability
    p_rf = rf_exp["predicted_prob"] or 0.5
    p_xgb = xgb_exp["predicted_prob"] or 0.5
    comb_prob = w_rf * p_rf + w_xgb * p_xgb

    # Combine all feature contributions linearly
    rf_map = {c["feature"]: c["shap_value"] for c in rf_exp["all_contributions"]}
    xgb_map = {c["feature"]: c["shap_value"] for c in xgb_exp["all_contributions"]}

    comb_contributions = []
    for i, name in enumerate(feature_names):
        val_rf = rf_map.get(name, 0.0)
        val_xgb = xgb_map.get(name, 0.0)
        ens_shap = w_rf * val_rf + w_xgb * val_xgb
        comb_contributions.append({
            "feature": name,
            "feature_index": i,
            "value": round(float(x_1d[i]), 4),
            "shap_value": round(ens_shap, 5),
            "rf_shap": round(val_rf, 5),
            "xgb_shap": round(val_xgb, 5),
            "abs_shap": abs(ens_shap),
            "direction": "Increases Late Risk" if ens_shap > 0 else "Reduces Late Risk",
        })

    # Top drivers
    comb_pos = sorted(
        [c for c in comb_contributions if c["shap_value"] > 0],
        key=lambda x: x["shap_value"],
        reverse=True,
    )[:top_k]

    comb_neg = sorted(
        [c for c in comb_contributions if c["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:top_k]

    return {
        "ensemble_name": "Weighted Soft Voting (0.70 RF + 0.30 XGB)",
        "weights": {"Random Forest": w_rf, "XGBoost": w_xgb},
        "base_value": round(comb_base, 4),
        "predicted_prob": round(comb_prob, 4),
        "rf_probability": round(p_rf, 4),
        "xgb_probability": round(p_xgb, 4),
        "top_positive": comb_pos,
        "top_negative": comb_neg,
        "all_contributions": sorted(comb_contributions, key=lambda x: x["abs_shap"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Business Domain Explanations & Translations
# ---------------------------------------------------------------------------

DOMAIN_INTERPRETATIONS = {
    "Days for shipment (scheduled)": (
        "Scheduled transit time allotted for the order. Tighter shipping deadlines "
        "(e.g., 1-2 days) give carriers minimal margin for error and substantially "
        "escalate late delivery risk."
    ),
    "Shipping Mode": (
        "Selected logistics fulfillment tier (Standard Class, Second Class, First Class, "
        "Same Day). Priority shipping modes carry stricter time constraints and higher "
        "failure rates under logistical bottlenecks."
    ),
    "Order Status": (
        "Processing state of the transaction (Complete, Pending, Processing, On Hold, etc.). "
        "Orders delayed during payment authorization or warehouse fulfillment carry higher risk."
    ),
    "Type": (
        "Payment and fulfillment method (DEBIT, TRANSFER, PAYMENT, CASH). Bank transfers and "
        "unsettled payments frequently delay shipment release."
    ),
    "Order City": (
        "Destination city of the delivery. Geographic distance, local routing infrastructure, "
        "and regional carrier coverage directly influence on-time delivery rates."
    ),
    "Order State": (
        "Destination administrative state / province. Regional delivery hubs vary in "
        "throughput efficiency and weather disruptions."
    ),
    "Customer State": (
        "Origin / billing location of the customer, often reflecting delivery corridor distances."
    ),
    "Order Customer Id": (
        "Customer profile indicator. Repeat corporate buyers often order predictable batches, "
        "whereas new or sporadic customers may have custom logistics requirements."
    ),
    "Order Item Discount Rate": (
        "Promotional discount applied to the item. Heavy discount events coincide with flash sales "
        "and warehouse order surges that overwhelm dispatch centers."
    ),
    "Product Category Id": (
        "Merchandise category. Bulky or high-demand goods require specialized handling and packaging."
    ),
}


def get_domain_context(feature_name: str) -> str:
    """Return supply chain business context for a given feature."""
    return DOMAIN_INTERPRETATIONS.get(
        feature_name,
        "Supply chain operational metric influencing logistics routing and fulfillment duration."
    )
