#!/usr/bin/env python3
import os, sys, pickle, json
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# — point this at your models folder —
HERE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, 'models')

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)

models = {
    '1w': load_model('breakout_model_1w.pkl'),
    '1m': load_model('breakout_model_1m.pkl'),
    '3m': load_model('breakout_model_3m.pkl'),
}

def evaluate_horizon(h):
    feat_f = os.path.join(MODELS_DIR, f"test_features_{h}.csv")
    labl_f = os.path.join(MODELS_DIR, f"test_labels_{h}.csv")

    if not os.path.exists(feat_f) or not os.path.exists(labl_f):
        print(f"\n❌ Missing test files for horizon '{h}'.")
        print(f"   → Looking for:\n     {feat_f}\n     {labl_f}")
        print("   Run your training script with test‐CSV export turned on, then try again.\n")
        return

    X_test = pd.read_csv(feat_f).values
    y_test = pd.read_csv(labl_f)['label'].values

    m       = models[h]
    y_pred  = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:,1]

    print(f"\n=== {h} Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=['HOLD','BUY']))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba),4))

if __name__ == '__main__':
    for horizon in ['1w','1m','3m']:
        evaluate_horizon(horizon)
