import json
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

rf = joblib.load('models/random_forest.pkl')
X_train = np.load('outputs/X_train.npy')
X_test = np.load('outputs/X_test.npy')

with open('outputs/preprocessing_metadata.json') as f:
    meta = json.load(f)
feat_names = meta['feature_names_encoded']

explainer = LimeTabularExplainer(
    training_data=X_train[:200],
    feature_names=feat_names,
    class_names=['On Time', 'Late'],
    mode='classification',
    random_state=42
)

exp = explainer.explain_instance(
    data_row=X_test[0],
    predict_fn=rf.predict_proba,
    num_features=8
)

print('LIME success! Top rules and weights:')
for rule, weight in exp.as_list():
    print(f'  {rule}: {weight:+.4f}')

