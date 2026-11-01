"""
src/preprocessing.py
--------------------
PURPOSE:
    Transforms the raw DataCo Smart Supply Chain DataFrame into a clean,
    model-ready format.  Implements leakage prevention, missing-value
    imputation, categorical encoding, numerical scaling, and a reproducible
    stratified train/test split.

PHASE: 3 — Data Preprocessing, Feature Engineering, and Leakage Prevention
STATUS: Implemented.

DESIGN PRINCIPLES:
    1. Split first, transform second.
       Imputers, encoders, and scalers are fitted ONLY on training data.
       The fitted objects are then applied to the test set to avoid
       data leakage from the test set into the preprocessing step.

    2. Transparent column decisions.
       Every column that is excluded is recorded in a review table with
       an explicit reason.  Nothing is silently removed.

    3. Leakage prevention.
       Columns that reveal the delivery outcome or that would not be
       available at prediction time are excluded.

    4. Reusability.
       The pipeline is saved to disk so that Phase 4 (model training)
       and Phase 6 (SHAP/LIME) can transform new data consistently.

PUBLIC API:
    run_preprocessing(df)  -> PreprocessingResult (namedtuple)
    get_status()           -> dict

ARTIFACTS SAVED (under outputs/):
    preprocessing_pipeline.pkl  — fitted ColumnTransformer pipeline
    preprocessing_metadata.json — column decisions, split info, feature names
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLUMN  = "Late_delivery_risk"
TEST_SIZE      = 0.30       # 30 % test, 70 % training
RANDOM_STATE   = 42
OUTPUTS_DIR    = "outputs"

# ---- Leakage columns -------------------------------------------------------
# These columns encode the delivery OUTCOME — they are generated AFTER the
# delivery decision is made and must never be used as predictive features.
# Using them would create a model that simply reads back the answer, not one
# that predicts the future.
LEAKAGE_COLS = {
    "Delivery Status":            "Directly encodes whether delivery was late/advance/on-time. Generated after delivery.",
    "Days for shipping (real)":   "Actual days taken — only known after delivery is complete.",
    "Late_delivery_risk":         "This is the target variable itself; excluded from features.",
}

# ---- Identifier / PII columns ----------------------------------------------
# These carry no predictive signal and some contain personally identifiable
# information (PII).  They are dropped from the feature matrix.
ID_COLS = {
    "Order Id":           "Unique order identifier — no predictive value.",
    "Order Item Id":      "Unique order-item identifier — no predictive value.",
    "Customer Id":        "Unique customer identifier — no predictive value.",
    "Customer Email":     "PII — email address, no predictive value.",
    "Customer Password":  "PII — password field, no predictive value.",
    "Product Description":"Free text, extremely high cardinality, no reliable signal.",
    "Order Zipcode":      "Mostly empty in this dataset; treated as identifier.",
}

# ---- High-cardinality / free-text columns ----------------------------------
# These have so many unique values that one-hot encoding would be impractical
# and label encoding would introduce a spurious ordinal relationship.
HIGH_CARDINALITY_COLS = {
    "Customer Fname":     "First name — very high cardinality, no predictive signal.",
    "Customer Lname":     "Last name — very high cardinality, no predictive signal.",
    "Customer Street":    "Street address — extremely high cardinality.",
    "Product Name":       "Product name — high cardinality; product category is used instead.",
    "Customer Full Name": "Duplicate of first + last name fields.",
}

# ---- Date / timestamp columns ----------------------------------------------
# Raw date strings are dropped; date-derived features could be engineered
# in a future phase if needed.
DATE_COLS = {
    "order date (DateOrders)": "Raw date string — not encoded; would require feature engineering.",
    "shipping date (DateOrders)":"Raw date string — also encodes delivery information.",
}

# All columns excluded in one consolidated dict  {column_name: reason}
ALL_EXCLUDED: dict = {}
ALL_EXCLUDED.update({k: f"[Leakage] {v}"           for k, v in LEAKAGE_COLS.items()})
ALL_EXCLUDED.update({k: f"[Identifier/PII] {v}"    for k, v in ID_COLS.items()})
ALL_EXCLUDED.update({k: f"[High cardinality] {v}"  for k, v in HIGH_CARDINALITY_COLS.items()})
ALL_EXCLUDED.update({k: f"[Date/timestamp] {v}"    for k, v in DATE_COLS.items()})

# Threshold: drop a column if more than this fraction of values are missing
MISSING_DROP_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Named return type — keeps the caller code readable
# ---------------------------------------------------------------------------

_PreprocessingResultBase = namedtuple(
    "PreprocessingResult",
    [
        "X_train",          # pd.DataFrame — encoded + scaled training features
        "X_test",           # pd.DataFrame — encoded + scaled test features
        "y_train",          # pd.Series    — training target
        "y_test",           # pd.Series    — test target
        "feature_names",    # list[str]    — feature column names after encoding
        "pipeline",         # fitted ColumnTransformer
        "metadata",         # dict         — full audit trail of decisions
        "errors",           # list[str]    — non-fatal warnings / issues found
    ],
)


class PreprocessingResult(_PreprocessingResultBase):
    """
    Preprocessing pipeline result container.

    Inherits from namedtuple for positional unpacking and dot-attribute access.
    Implements string-key indexing and get() for backward compatibility with dictionary callers.
    """

    def __getitem__(self, item):
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __contains__(self, key):
        return hasattr(self, key) or super().__contains__(key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _identify_drop_cols(df: pd.DataFrame) -> dict:
    """
    Build the full dict of columns to drop, considering:
      - Pre-configured exclusion lists (leakage, IDs, dates, high cardinality)
      - Columns that are actually present in this specific DataFrame
      - Columns with excessive missing values (> MISSING_DROP_THRESHOLD)
      - Constant columns (only one unique value)

    Returns {column_name: reason_string}
    """
    to_drop: dict = {}

    # 1. Pre-configured columns — only drop those that actually exist
    for col, reason in ALL_EXCLUDED.items():
        if col in df.columns:
            to_drop[col] = reason

    # 2. Columns with excessive missing values
    for col in df.columns:
        if col in to_drop:
            continue
        missing_frac = df[col].isnull().mean()
        if missing_frac > MISSING_DROP_THRESHOLD:
            to_drop[col] = (
                f"[Excessive missing] {missing_frac*100:.1f}% of values are missing "
                f"(threshold: {MISSING_DROP_THRESHOLD*100:.0f}%)."
            )

    # 3. Constant columns (zero variance — useless for any model)
    for col in df.columns:
        if col in to_drop or col == TARGET_COLUMN:
            continue
        if df[col].nunique(dropna=True) <= 1:
            to_drop[col] = "[Constant] Column has only one unique value — no predictive value."

    return to_drop


def _build_column_review(df: pd.DataFrame, drop_dict: dict,
                         kept_cols: list) -> pd.DataFrame:
    """
    Return a DataFrame suitable for display in the Preprocessing page.

    Columns: Column Name | Type | Decision | Reason
    """
    rows = []
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        dtype_str = str(df[col].dtype)
        if col in drop_dict:
            decision = "Excluded"
            reason   = drop_dict[col]
        elif col in kept_cols:
            decision = "Included"
            reason   = "No exclusion criterion met."
        else:
            decision = "Unknown"
            reason   = ""
        rows.append({
            "Column":   col,
            "Dtype":    dtype_str,
            "Decision": decision,
            "Reason":   reason,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_preprocessing(
    df: pd.DataFrame,
    save_artifacts: bool = True,
    output_dir: str = OUTPUTS_DIR,
) -> PreprocessingResult:
    """
    Run the complete Phase 3 preprocessing pipeline on the loaded DataFrame.

    Steps
    -----
    1.  Validate the target column.
    2.  Identify and record all columns to be excluded (with reasons).
    3.  Build the feature matrix X and target vector y.
    4.  Stratified train/test split (70/30, random_state=42).
    5.  Identify numerical and categorical features in X_train.
    6.  Build a ColumnTransformer pipeline:
          - Numerical:   median imputation → StandardScaler
          - Categorical: constant imputation ('missing') → OrdinalEncoder
              (OrdinalEncoder handles unseen categories at inference time)
    7.  Fit the pipeline on X_train only.
    8.  Transform X_train and X_test.
    9.  Validate that the target is absent from the feature matrix.
    10. Save the pipeline and metadata to outputs/.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from data_loader.load_dataset().

    Returns
    -------
    PreprocessingResult namedtuple — see field definitions above.

    Raises
    ------
    ValueError
        If the target column is absent or the dataset is too small to split.
    """
    errors: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Target validation
    # ------------------------------------------------------------------
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            "Load the correct dataset before running preprocessing."
        )

    # Check target dtype and unique values
    target_series  = df[TARGET_COLUMN].dropna()
    unique_vals    = sorted(target_series.unique().tolist())
    n_target_null  = int(df[TARGET_COLUMN].isnull().sum())

    if n_target_null > 0:
        errors.append(
            f"{n_target_null:,} rows have a missing target value and will be dropped."
        )
        df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Confirm binary classification
    if set(unique_vals) - {0, 1}:
        # Try casting — the DataCo target is stored as int64 but let's be safe
        try:
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
            unique_vals = sorted(df[TARGET_COLUMN].unique().tolist())
        except Exception:
            pass

    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains unexpected values: {unique_vals}. "
            "Expected binary labels 0 and 1 for binary classification."
        )

    target_value_counts = df[TARGET_COLUMN].value_counts().sort_index().to_dict()

    # ------------------------------------------------------------------
    # Step 2: Identify columns to drop and build feature matrix
    # ------------------------------------------------------------------
    drop_dict  = _identify_drop_cols(df)
    cols_to_drop = list(drop_dict.keys())

    kept_cols  = [
        c for c in df.columns
        if c != TARGET_COLUMN and c not in cols_to_drop
    ]

    X = df[kept_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Sanity check: target must not be in X
    assert TARGET_COLUMN not in X.columns, "FATAL: target leaked into feature matrix."

    if X.shape[1] == 0:
        raise ValueError("No features remain after exclusions. Check the dataset columns.")

    column_review_df = _build_column_review(df, drop_dict, kept_cols)

    # ------------------------------------------------------------------
    # Step 3: Stratified train/test split
    # ------------------------------------------------------------------
    if len(X) < 10:
        raise ValueError("Dataset is too small to split (fewer than 10 rows).")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # maintain class proportions in both splits
    )

    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)

    train_class_dist = y_train.value_counts().sort_index().to_dict()
    test_class_dist  = y_test.value_counts().sort_index().to_dict()

    # ------------------------------------------------------------------
    # Step 4: Identify feature types (based on TRAINING data only)
    # ------------------------------------------------------------------
    num_cols  = X_train_raw.select_dtypes(include="number").columns.tolist()
    cat_cols  = X_train_raw.select_dtypes(exclude="number").columns.tolist()

    # ------------------------------------------------------------------
    # Step 5: Build the sklearn ColumnTransformer pipeline
    #
    # Numerical pipeline:
    #   - SimpleImputer(strategy="median")  — robust to outliers
    #   - StandardScaler()                  — required for LR, SVM, DNN
    #
    # Categorical pipeline:
    #   - SimpleImputer(strategy="constant", fill_value="missing")
    #   - OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    #       Assigns -1 to any category seen at test time but not at train time.
    #       Tree-based models (DT, RF, XGBoost) can handle this naturally.
    #       For LR and SVM we will apply one-hot encoding in a later step if
    #       needed, but OrdinalEncoder is the safe default here.
    #
    # remainder="drop" ensures any column not in num_cols or cat_cols
    # (e.g., a stray all-null column) is silently dropped.
    # ------------------------------------------------------------------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = []
    if num_cols:
        transformers.append(("numerical", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("categorical", cat_pipeline, cat_cols))

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,   # keep original column names
    )

    # ------------------------------------------------------------------
    # Step 6: Fit on TRAINING data only, then transform both sets
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train_arr = pipeline.fit_transform(X_train_raw)
        X_test_arr  = pipeline.transform(X_test_raw)

    # Reconstruct as DataFrames with named columns
    try:
        feature_names = list(pipeline.get_feature_names_out())
    except Exception:
        feature_names = num_cols + cat_cols

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    # ------------------------------------------------------------------
    # Step 7: Validation checks
    # ------------------------------------------------------------------
    validation_checks = {
        "target_not_in_features":      TARGET_COLUMN not in X_train.columns,
        "train_test_schema_match":     list(X_train.columns) == list(X_test.columns),
        "no_inf_in_train":             not np.isinf(X_train.values).any(),
        "no_inf_in_test":              not np.isinf(X_test.values).any(),
        "train_rows_gt_zero":          len(X_train) > 0,
        "test_rows_gt_zero":           len(X_test)  > 0,
        "stratification_confirmed":    True,   # train_test_split was called with stratify=y
    }

    for check_name, passed in validation_checks.items():
        if not passed:
            errors.append(f"Validation failed: {check_name}")

    # ------------------------------------------------------------------
    # Step 8: Build metadata record
    # ------------------------------------------------------------------
    metadata = {
        "target_column":         TARGET_COLUMN,
        "original_shape":        list(df.shape),
        "n_rows_after_drop":     len(df),
        "n_target_null_dropped": n_target_null,
        "target_value_counts":   {str(k): int(v) for k, v in target_value_counts.items()},
        "kept_columns":          kept_cols,
        "excluded_columns":      {k: v for k, v in drop_dict.items()},
        "n_features_kept":       len(kept_cols),
        "n_features_excluded":   len(cols_to_drop),
        "num_features":          num_cols,
        "cat_features":          cat_cols,
        "feature_names_encoded": feature_names,
        "test_size":             TEST_SIZE,
        "random_state":          RANDOM_STATE,
        "stratified":            True,
        "n_train":               len(X_train),
        "n_test":                len(X_test),
        "train_class_dist":      {str(k): int(v) for k, v in train_class_dist.items()},
        "test_class_dist":       {str(k): int(v) for k, v in test_class_dist.items()},
        "missing_strategy_num":  "median imputation then StandardScaler",
        "missing_strategy_cat":  "constant fill ('missing') then OrdinalEncoder",
        "unseen_category_value": -1,
        "validation_checks":     validation_checks,
        "errors":                errors,
    }

    # ------------------------------------------------------------------
    # Step 9: Save artefacts to outputs/
    #
    # DataFrames and Series are saved as numpy .npz arrays (no pyarrow
    # dependency) alongside a separate JSON for column names.
    # The pipeline is saved with joblib (sklearn objects are safe).
    # ------------------------------------------------------------------
    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.pkl")
        metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")

        joblib.dump(pipeline, pipeline_path, compress=3)

        # Save X_train, X_test as numpy arrays + column-name JSON
        np.save(os.path.join(output_dir, "X_train.npy"), X_train.values)
        np.save(os.path.join(output_dir, "X_test.npy"),  X_test.values)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
        np.save(os.path.join(output_dir, "y_test.npy"),  y_test.values)

        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return PreprocessingResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        pipeline=pipeline,
        metadata=metadata,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Artefact loader (used by Phase 4)
# ---------------------------------------------------------------------------

def load_preprocessing_artifacts(output_dir: str = OUTPUTS_DIR) -> PreprocessingResult | None:
    """
    Load saved preprocessing artefacts from outputs/ (or custom directory).

    Supports both the numpy .npy format (current) and legacy joblib .pkl
    format so that existing outputs/ directories work without rerunning.

    Returns None if artefacts do not exist yet.

    Returns
    -------
    PreprocessingResult | None
        Namedtuple with fields (X_train, X_test, y_train, y_test, feature_names,
        pipeline, metadata, errors). Supports dot access, tuple indexing/unpacking,
        and dictionary-style string keys.
    """
    meta_path     = os.path.join(output_dir, "preprocessing_metadata.json")
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.pkl")

    if not os.path.isfile(meta_path) or not os.path.isfile(pipeline_path):
        return None

    with open(meta_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    feature_names = metadata.get("feature_names_encoded", [])

    # --- Try numpy .npy format first (current) ---
    npy_paths = {
        "X_train": os.path.join(output_dir, "X_train.npy"),
        "X_test":  os.path.join(output_dir, "X_test.npy"),
        "y_train": os.path.join(output_dir, "y_train.npy"),
        "y_test":  os.path.join(output_dir, "y_test.npy"),
    }

    if all(os.path.isfile(p) for p in npy_paths.values()):
        X_train_arr = np.load(npy_paths["X_train"], allow_pickle=False)
        X_test_arr  = np.load(npy_paths["X_test"],  allow_pickle=False)
        y_train_arr = np.load(npy_paths["y_train"], allow_pickle=False)
        y_test_arr  = np.load(npy_paths["y_test"],  allow_pickle=False)

        X_train = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)
        y_train = pd.Series(y_train_arr, name=TARGET_COLUMN)
        y_test  = pd.Series(y_test_arr,  name=TARGET_COLUMN)

        return PreprocessingResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            pipeline=joblib.load(pipeline_path),
            metadata=metadata,
            errors=[],
        )

    # --- Fall back to legacy joblib .pkl format ---
    pkl_paths = {
        "X_train": os.path.join(output_dir, "X_train.pkl"),
        "X_test":  os.path.join(output_dir, "X_test.pkl"),
        "y_train": os.path.join(output_dir, "y_train.pkl"),
        "y_test":  os.path.join(output_dir, "y_test.pkl"),
    }

    if all(os.path.isfile(p) for p in pkl_paths.values()):
        try:
            pkl_X_train = joblib.load(pkl_paths["X_train"])
            pkl_X_test  = joblib.load(pkl_paths["X_test"])
            pkl_y_train = joblib.load(pkl_paths["y_train"])
            pkl_y_test  = joblib.load(pkl_paths["y_test"])
            # Ensure DataFrames have column names
            if isinstance(pkl_X_train, np.ndarray):
                pkl_X_train = pd.DataFrame(pkl_X_train, columns=feature_names)
            if isinstance(pkl_X_test, np.ndarray):
                pkl_X_test  = pd.DataFrame(pkl_X_test,  columns=feature_names)
            if isinstance(pkl_y_train, np.ndarray):
                pkl_y_train = pd.Series(pkl_y_train, name=TARGET_COLUMN)
            if isinstance(pkl_y_test, np.ndarray):
                pkl_y_test  = pd.Series(pkl_y_test,  name=TARGET_COLUMN)
            return PreprocessingResult(
                X_train=pkl_X_train,
                X_test=pkl_X_test,
                y_train=pkl_y_train,
                y_test=pkl_y_test,
                feature_names=feature_names,
                pipeline=joblib.load(pipeline_path),
                metadata=metadata,
                errors=[],
            )
        except Exception:
            pass  # load error — fall through

    return None


# ---------------------------------------------------------------------------
# Module status (used by app.py Home page)
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """
    Return the readiness status of Phase 3 for the Home page status cards.

    Checks whether preprocessing artefacts have been saved to outputs/.
    Does NOT re-run preprocessing — that is triggered from the Streamlit page.
    """
    # Check only the metadata file — avoids loading heavy numpy arrays
    meta_path = os.path.join(OUTPUTS_DIR, "preprocessing_metadata.json")
    npy_ready = all(
        os.path.isfile(os.path.join(OUTPUTS_DIR, f))
        for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
                  "preprocessing_pipeline.pkl"]
    )

    if not os.path.isfile(meta_path):
        return {
            "module":  "preprocessing",
            "phase":   3,
            "ready":   False,
            "message": (
                "No preprocessing artefacts found. "
                "Load a dataset and run preprocessing on the Preprocessing page."
            ),
        }

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return {
            "module":  "preprocessing",
            "phase":   3,
            "ready":   npy_ready,
            "message": (
                f"Preprocessing complete. "
                f"{meta['n_train']:,} training rows, {meta['n_test']:,} test rows, "
                f"{len(meta['feature_names_encoded'])} features."
                if npy_ready
                else "Preprocessing metadata found but data files are missing. Re-run preprocessing."
            ),
        }
    except Exception:
        return {
            "module":  "preprocessing",
            "phase":   3,
            "ready":   False,
            "message": "Could not read preprocessing metadata. Re-run preprocessing.",
        }
