# ml_model.py
import os, json, pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# load everything
scaler     = pickle.load(open(os.path.join(MODELS_DIR,'scaler.pkl'),'rb'))
models     = {h: pickle.load(open(os.path.join(MODELS_DIR,f'breakout_model_{h}.pkl'),'rb'))
              for h in ['1w','1m','3m']}
thresholds = json.load(open(os.path.join(MODELS_DIR,'thresholds.json'),'r'))

def predict_breakouts(open_, high, low, close):
    """Return dict with score, decision, target for each horizon."""
    feats = np.array([open_, high, low, close]).reshape(1,4)
    feats_scaled = scaler.transform(feats)
    out = {}
    for h, clf in models.items():
        p = float(clf.predict_proba(feats_scaled)[0][1])
        d = 'BUY' if p>=thresholds[h] else 'HOLD'
        out[h] = {
            'score': round(p,4),
            'decision': d,
            'target': round(close*(1+thresholds[h]),2)
        }
    return out
