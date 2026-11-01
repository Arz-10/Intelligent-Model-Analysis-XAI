import time
import json
import joblib
import numpy as np
import shap

print("Loading artifacts...")
rf = joblib.load("models/random_forest.pkl")
X_test = np.load("outputs/X_test.npy")
with open("outputs/preprocessing_metadata.json") as f:
    meta = json.load(f)
feat_names = meta["feature_names_encoded"]

n_samples = 250
print(f"Computing TreeExplainer on {n_samples} samples...")
t0 = time.time()
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(X_test[:n_samples])
elapsed = time.time() - t0
print(f"Computed in {elapsed:.2f} seconds.")

# Process shape
if isinstance(sv, list):
    vals = sv[1]
elif getattr(sv, 'ndim', 0) == 3:
    vals = sv[:, :, 1]
else:
    vals = sv

mean_abs = np.mean(np.abs(vals), axis=0)
order = np.argsort(mean_abs)[::-1]

print("Top 10 features by mean absolute SHAP:")
for rank, idx in enumerate(order[:10], 1):
    print(f"{rank:2d}. {feat_names[idx]:35s} | Mean |SHAP|: {mean_abs[idx]:.4f}")

