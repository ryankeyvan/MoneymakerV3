# evaluate.py
import os, pickle, json
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
models = {h: pickle.load(open(os.path.join(MODELS_DIR,f'breakout_model_{h}.pkl'),'rb'))
          for h in ['1w','1m','3m']}

def evaluate_horizon(h):
    X = pd.read_csv(f'{MODELS_DIR}/test_features_{h}.csv').values
    y = pd.read_csv(f'{MODELS_DIR}/test_labels_{h}.csv')['label'].values
    m = models[h]
    y_pred = m.predict(X)
    y_proba = m.predict_proba(X)[:,1]
    print(f"\n=== {h} Performance ===")
    print(classification_report(y, y_pred, target_names=['HOLD','BUY']))
    print("ROC AUC:", round(roc_auc_score(y, y_proba),4))

if __name__=='__main__':
    for h in ['1w','1m','3m']:
        evaluate_horizon(h)
