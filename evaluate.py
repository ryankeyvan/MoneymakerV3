# evaluate.py
import os, pickle, json
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# adjust to wherever your models live
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model(name):
    with open(os.path.join(MODELS_DIR, name), 'rb') as f:
        return pickle.load(f)

models = {
    '1w': load_model('breakout_model_1w.pkl'),
    '1m': load_model('breakout_model_1m.pkl'),
    '3m': load_model('breakout_model_3m.pkl'),
}

def evaluate_horizon(h):
    X_test = pd.read_csv(f'{MODELS_DIR}/test_features_{h}.csv').values
    y_test = pd.read_csv(f'{MODELS_DIR}/test_labels_{h}.csv')['label'].values

    m = models[h]
    y_pred  = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:,1]

    print(f"\n=== {h} Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=['HOLD','BUY']))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba),4))

if __name__ == '__main__':
    for horizon in ['1w','1m','3m']:
        evaluate_horizon(horizon)
