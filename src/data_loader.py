"""
src/data_loader.py
------------------
PURPOSE:
    Loads the DataCo Smart Supply Chain CSV dataset and produces a detailed
    validation report.  All analysis is performed on the actual data that is
    passed in — nothing is fabricated.

PHASE: 2 (Dataset Loading & Validation)
STATUS: Implemented.

PUBLIC API:
    find_dataset_file(data_dir)  -> str | None
    load_dataset(filepath)       -> tuple[pd.DataFrame, dict]
    build_report(df)             -> dict
    get_status()                 -> dict

NOTES:
    - The DataCo CSV uses latin-1 encoding (contains accented characters).
    - Column names are verified from the actual file; nothing is assumed.
    - Target column: 'Late_delivery_risk' (verified, not assumed present).
    - Columns flagged as potential leakage sources are identified by pattern
      matching only; no columns are removed here — that is Phase 3's job.
"""

import os
import io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "data"
TARGET_COLUMN    = "Late_delivery_risk"

# Column name fragments that may indicate direct leakage or non-predictive
# identifiers.  These are flagged for human review, NOT automatically removed.
LEAKAGE_PATTERNS = [
    "delivery status",
    "late delivery",
    "order status",
    "shipping date",
    "late_delivery_risk",   # the target itself
]

ID_PATTERNS = [
    "id",
    "email",
    "password",
    "zipcode",
    "zip code",
    "phone",
]


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def find_dataset_file(data_dir: str = DEFAULT_DATA_DIR) -> str | None:
    """
    Scan *data_dir* for the first CSV file and return its full path.

    Parameters
    ----------
    data_dir : str
        Directory to search.

    Returns
    -------
    str | None
        Absolute or relative path to the first .csv found, or None.
    """
    if not os.path.isdir(data_dir):
        return None
    for filename in sorted(os.listdir(data_dir)):
        if filename.lower().endswith(".csv"):
            return os.path.join(data_dir, filename)
    return None


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def _try_read_csv(source) -> tuple[pd.DataFrame | None, str | None]:
    """
    Attempt to read a CSV from a file path or a file-like object.

    Tries latin-1 first (required for the DataCo file), then utf-8.
    Returns (DataFrame, None) on success or (None, error_message) on failure.
    """
    for encoding in ("latin-1", "utf-8", "cp1252"):
        try:
            if isinstance(source, str):
                df = pd.read_csv(source, encoding=encoding, low_memory=False)
            else:
                source.seek(0)                          # reset file pointer
                df = pd.read_csv(source, encoding=encoding, low_memory=False)
            return df, None
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return None, str(exc)
    return None, "Could not decode file with latin-1, utf-8, or cp1252 encoding."


# ---------------------------------------------------------------------------
# Validation Report Builder
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, source_name: str = "") -> dict:
    """
    Build a comprehensive validation report from a loaded DataFrame.

    All values in the returned dict are derived from the real data.

    Parameters
    ----------
    df          : pd.DataFrame  — the loaded dataset
    source_name : str           — display name of the source file

    Returns
    -------
    dict with keys:
        success, source_name, shape, n_rows, n_cols,
        columns, dtypes, numerical_cols, categorical_cols,
        duplicate_rows, total_missing, missing_by_column,
        target_found, target_value_counts, target_missing,
        leakage_candidates, id_candidates,
        errors, warnings
    """
    report: dict = {
        "success":             True,
        "source_name":         source_name,
        "shape":               df.shape,
        "n_rows":              len(df),
        "n_cols":              len(df.columns),
        "columns":             list(df.columns),
        "dtypes":              df.dtypes.astype(str).to_dict(),
        "numerical_cols":      list(df.select_dtypes(include="number").columns),
        "categorical_cols":    list(df.select_dtypes(exclude="number").columns),
        "duplicate_rows":      int(df.duplicated().sum()),
        "total_missing":       int(df.isnull().sum().sum()),
        "missing_by_column":   {},
        "target_found":        False,
        "target_value_counts": {},
        "target_missing":      0,
        "leakage_candidates":  [],
        "id_candidates":       [],
        "errors":              [],
        "warnings":            [],
    }

    # --- Missing values per column ---
    missing_counts  = df.isnull().sum()
    missing_pct     = (missing_counts / len(df) * 100).round(2)
    report["missing_by_column"] = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing_counts[col] > 0
    }

    # --- Target column check ---
    if TARGET_COLUMN in df.columns:
        report["target_found"]        = True
        report["target_missing"]      = int(df[TARGET_COLUMN].isnull().sum())
        report["target_value_counts"] = df[TARGET_COLUMN].value_counts().to_dict()
    else:
        report["warnings"].append(
            f"Target column '{TARGET_COLUMN}' was not found in this CSV. "
            "You will need to identify the correct target column before "
            "model training can begin."
        )

    # --- Potential leakage / ID column flagging (name-based heuristic only) ---
    cols_lower = {col: col.lower() for col in df.columns}

    for col, col_l in cols_lower.items():
        if col == TARGET_COLUMN:
            continue                                    # target itself is handled above
        if any(pat in col_l for pat in LEAKAGE_PATTERNS):
            report["leakage_candidates"].append(col)
        elif any(pat in col_l for pat in ID_PATTERNS):
            report["id_candidates"].append(col)

    return report


# ---------------------------------------------------------------------------
# Main Load Function
# ---------------------------------------------------------------------------

def load_dataset(source, source_name: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV and return (DataFrame, report).

    Parameters
    ----------
    source      : str or file-like object
                  A file path string  OR  a Streamlit UploadedFile / BytesIO.
    source_name : str
                  Human-readable name to display (e.g. the uploaded filename).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        df     — loaded DataFrame (empty on failure)
        report — validation report dict; check report['success'] first.
    """
    error_report = {
        "success":      False,
        "source_name":  source_name,
        "errors":       [],
        "warnings":     [],
    }

    # --- Resolve source name ---
    if not source_name and isinstance(source, str):
        source_name = os.path.basename(source)

    # --- Guard: path must exist when a string is given ---
    if isinstance(source, str) and not os.path.isfile(source):
        error_report["errors"].append(f"File not found: {source}")
        return pd.DataFrame(), error_report

    # --- Read the CSV ---
    df, read_error = _try_read_csv(source)
    if df is None:
        error_report["errors"].append(f"Could not read CSV: {read_error}")
        return pd.DataFrame(), error_report

    if df.empty:
        error_report["errors"].append("The CSV file is empty.")
        return pd.DataFrame(), error_report

    # --- Build and return the full report ---
    report = build_report(df, source_name=source_name)
    return df, report


# ---------------------------------------------------------------------------
# Module Status (used by app.py Home page)
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """
    Return whether a CSV file exists in the data/ directory.

    Used by the Streamlit dashboard to populate the Phase 2 status card.
    This does NOT load the file — it only checks for its presence.
    """
    csv_path = find_dataset_file()
    return {
        "module":        "data_loader",
        "phase":         2,
        "ready":         csv_path is not None,
        "dataset_found": csv_path,
        "message": (
            f"Dataset file detected: {csv_path}"
            if csv_path
            else "No CSV file found in the data/ folder. "
                 "Upload a file on the Dataset page to proceed."
        ),
    }
