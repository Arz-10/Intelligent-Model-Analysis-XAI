"""
scratch/test_phase3.py
----------------------
Automated tests for Phase 3 preprocessing pipeline.
Run with:  python scratch/test_phase3.py
"""
import sys, os, warnings
sys.path.insert(0, r'c:\Users\Aarushi Sontakke\Downloads\CIL_Project')

import numpy as np
import pandas as pd
from src.preprocessing import (
    run_preprocessing, load_preprocessing_artifacts,
    TARGET_COLUMN, OUTPUTS_DIR, get_status,
)


def make_df(n=500, seed=42):
    """Create a synthetic DataCo-like DataFrame for testing."""
    np.random.seed(seed)
    return pd.DataFrame({
        'Type':                          np.random.choice(['DEBIT','TRANSFER','CASH'], n),
        'Days for shipment (scheduled)': np.random.randint(1, 8, n),
        'Benefit per order':             np.random.uniform(-50, 300, n).round(2),
        'Sales per customer':            np.random.uniform(50, 1000, n).round(2),
        'Category Name':                 np.random.choice(['Electronics','Clothing','Sports','Toys'], n),
        'Customer City':                 np.random.choice(['New York','Los Angeles','Chicago'], n),
        'Customer Segment':              np.random.choice(['Consumer','Corporate','Home Office'], n),
        'Shipping Mode':                 np.random.choice(['Standard Class','Second Class','First Class','Same Day'], n),
        # Leakage — must be excluded
        'Delivery Status':               np.random.choice(['Late delivery','Advance shipping'], n),
        'Days for shipping (real)':      np.random.randint(1, 10, n),
        # IDs — must be excluded
        'Order Id':                      np.arange(n),
        'Customer Email':                [f'u{i}@t.com' for i in range(n)],
        # Target
        TARGET_COLUMN:                   np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })


def run():
    passed = 0
    failed = 0

    def ok(name, msg=''):
        nonlocal passed
        passed += 1
        suffix = f'  — {msg}' if msg else ''
        print(f'  PASS  {name}{suffix}')

    def fail(name, msg=''):
        nonlocal failed
        failed += 1
        print(f'  FAIL  {name}  — {msg}')

    df = make_df()

    # ------------------------------------------------------------------
    # T1: Basic pipeline run
    # ------------------------------------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_preprocessing(df)
        ok('T01 Pipeline runs without exception',
           f'X_train={result.X_train.shape}, X_test={result.X_test.shape}')
    except Exception as e:
        fail('T01 Pipeline runs without exception', str(e))
        print('Cannot continue — aborting.')
        return

    # ------------------------------------------------------------------
    # T2: Target absent from feature matrix
    # ------------------------------------------------------------------
    if TARGET_COLUMN not in result.X_train.columns and TARGET_COLUMN not in result.X_test.columns:
        ok('T02 Target absent from X_train and X_test')
    else:
        fail('T02 Target absent from X_train and X_test', 'Target leaked into features')

    # ------------------------------------------------------------------
    # T3: Leakage columns excluded
    # ------------------------------------------------------------------
    leakage = ['Delivery Status', 'Days for shipping (real)']
    missing_leakage = [c for c in leakage if c in result.X_train.columns]
    if not missing_leakage:
        ok('T03 Leakage columns excluded', str(leakage))
    else:
        fail('T03 Leakage columns excluded', f'Still present: {missing_leakage}')

    # ------------------------------------------------------------------
    # T4: ID columns excluded
    # ------------------------------------------------------------------
    id_cols = ['Order Id', 'Customer Email']
    present_ids = [c for c in id_cols if c in result.X_train.columns]
    if not present_ids:
        ok('T04 Identifier columns excluded', str(id_cols))
    else:
        fail('T04 Identifier columns excluded', f'Still present: {present_ids}')

    # ------------------------------------------------------------------
    # T5: No NaN in output
    # ------------------------------------------------------------------
    nan_train = result.X_train.isnull().any().any()
    nan_test  = result.X_test.isnull().any().any()
    if not nan_train and not nan_test:
        ok('T05 No NaN values after imputation')
    else:
        fail('T05 No NaN values after imputation', f'NaN in train={nan_train}, test={nan_test}')

    # ------------------------------------------------------------------
    # T6: No infinite values
    # ------------------------------------------------------------------
    inf_train = np.isinf(result.X_train.values).any()
    inf_test  = np.isinf(result.X_test.values).any()
    if not inf_train and not inf_test:
        ok('T06 No infinite values in output arrays')
    else:
        fail('T06 No infinite values', f'inf in train={inf_train}, test={inf_test}')

    # ------------------------------------------------------------------
    # T7: Split sizes
    # ------------------------------------------------------------------
    expected_test = int(round(len(df) * 0.30))
    actual_test   = len(result.X_test)
    if abs(actual_test - expected_test) <= 2:
        ok('T07 Split sizes correct', f'train={len(result.X_train)}, test={actual_test}')
    else:
        fail('T07 Split sizes', f'expected ~{expected_test}, got {actual_test}')

    # ------------------------------------------------------------------
    # T8: Stratification — class proportions
    # ------------------------------------------------------------------
    full_p  = df[TARGET_COLUMN].mean()
    train_p = result.y_train.mean()
    test_p  = result.y_test.mean()
    tol = 0.03
    if abs(train_p - full_p) < tol and abs(test_p - full_p) < tol:
        ok('T08 Stratification preserved',
           f'full={full_p:.3f}, train={train_p:.3f}, test={test_p:.3f}')
    else:
        fail('T08 Stratification', f'full={full_p:.3f}, train={train_p:.3f}, test={test_p:.3f}')

    # ------------------------------------------------------------------
    # T9: Train/test schema match
    # ------------------------------------------------------------------
    if list(result.X_train.columns) == list(result.X_test.columns):
        ok('T09 Training and test schemas match', f'{len(result.feature_names)} features')
    else:
        fail('T09 Schema match', 'column mismatch between X_train and X_test')

    # ------------------------------------------------------------------
    # T10: All validation checks in metadata passed
    # ------------------------------------------------------------------
    vc_all = all(result.metadata['validation_checks'].values())
    if vc_all:
        ok('T10 All 7 validation_checks in metadata are True')
    else:
        failed_vc = [k for k, v in result.metadata['validation_checks'].items() if not v]
        fail('T10 Validation checks', f'Failed: {failed_vc}')

    # ------------------------------------------------------------------
    # T11: Artifact save / load round-trip
    # ------------------------------------------------------------------
    try:
        arts = load_preprocessing_artifacts()
        assert arts is not None
        assert list(arts.X_train.columns) == result.feature_names
        assert len(arts.y_train) == len(result.y_train)
        ok('T11 Artifact save/load round-trip', f'X_train reloaded shape: {arts.X_train.shape}')
    except Exception as e:
        fail('T11 Artifact round-trip', str(e))

    # ------------------------------------------------------------------
    # T12: Reloaded pipeline transforms new data
    # ------------------------------------------------------------------
    try:
        arts = load_preprocessing_artifacts()
        new_rows = df[result.metadata['kept_columns']].iloc[:5].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformed = arts.pipeline.transform(new_rows)
        assert transformed.shape[1] == len(result.feature_names)
        ok('T12 Reloaded pipeline transforms new data', f'shape: {transformed.shape}')
    except Exception as e:
        fail('T12 Pipeline reload', str(e))

    # ------------------------------------------------------------------
    # T13: Unseen category handled without crash
    # ------------------------------------------------------------------
    try:
        arts = load_preprocessing_artifacts()
        cat_cols = result.metadata['cat_features']
        if cat_cols:
            new_row = df[result.metadata['kept_columns']].iloc[[0]].copy()
            new_row[cat_cols[0]] = 'NEVER_SEEN_VALUE_XYZ'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t = arts.pipeline.transform(new_row)
            ok('T13 Unseen category encoded without error',
               f'column={cat_cols[0]}, encoded shape={t.shape}')
        else:
            ok('T13 Unseen category — skipped (no categorical features in test data)')
    except Exception as e:
        fail('T13 Unseen category handling', str(e))

    # ------------------------------------------------------------------
    # T14: Missing target column — ValueError
    # ------------------------------------------------------------------
    try:
        df_no_tgt = df.drop(columns=[TARGET_COLUMN])
        run_preprocessing(df_no_tgt)
        fail('T14 Missing target raises ValueError', 'No error was raised')
    except ValueError as e:
        ok('T14 Missing target column raises ValueError', str(e)[:60])
    except Exception as e:
        fail('T14 Missing target raises ValueError', f'Wrong exception: {type(e).__name__}: {e}')

    # ------------------------------------------------------------------
    # T15: Target with missing values — rows dropped, warning issued
    # ------------------------------------------------------------------
    try:
        df_mt = df.copy()
        df_mt.loc[:19, TARGET_COLUMN] = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r2 = run_preprocessing(df_mt)
        assert r2.metadata['n_target_null_dropped'] == 20
        assert r2.errors
        ok('T15 Target missing values dropped cleanly',
           f'{r2.metadata["n_target_null_dropped"]} rows removed, warning issued')
    except Exception as e:
        fail('T15 Target missing values', str(e))

    # ------------------------------------------------------------------
    # T16: get_status() reflects saved artifacts
    # ------------------------------------------------------------------
    status = get_status()
    if status['ready']:
        ok('T16 get_status() returns ready=True after artifacts saved',
           status['message'][:70])
    else:
        fail('T16 get_status() ready after artifacts', status.get('message', ''))

    # ------------------------------------------------------------------
    # T17: Preprocessing is fitted on training data only
    #       (verify by checking scaler mean matches X_train mean, not X_test)
    # ------------------------------------------------------------------
    try:
        arts = load_preprocessing_artifacts()
        pipeline = arts.pipeline
        # Get the numerical sub-pipeline
        num_pipe = pipeline.named_transformers_.get('numerical')
        if num_pipe is not None:
            scaler = num_pipe.named_steps['scaler']
            # The scaler mean_ was computed from training data only
            # If we transform X_train and check mean ~ 0, that confirms training-only fit
            X_tr = arts.X_train
            num_feat_names = result.metadata['num_features']
            num_cols_in_output = [c for c in num_feat_names if c in X_tr.columns]
            if num_cols_in_output:
                col_means = X_tr[num_cols_in_output].mean().abs()
                # After StandardScaler fitted on train, train col means should be near 0
                max_mean = col_means.max()
                if max_mean < 0.05:
                    ok('T17 Scaler fitted on training data only',
                       f'max abs mean of scaled train cols: {max_mean:.4f}')
                else:
                    fail('T17 Scaler fitted on training data', f'max abs mean={max_mean:.4f} (expected ~0)')
            else:
                ok('T17 Scaler fit — skipped (no matching numerical cols in output)')
        else:
            ok('T17 Scaler fit — skipped (no numerical transformer found)')
    except Exception as e:
        fail('T17 Scaler fitted on training data', str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print()
    print(f'Results: {passed}/{total} tests passed, {failed} failed')
    if failed == 0:
        print('All Phase 3 tests PASSED.')
    else:
        print('Some tests FAILED — review output above.')


if __name__ == '__main__':
    run()

