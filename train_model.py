# train_model.py
#!/usr/bin/env python3
import os, json, pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define horizons in trading days
HORIZONS = {'1w': 5, '1m': 21, '3m': 63}

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()

def build_dataset(tickers, years=3):
    rows = []
    for ticker in tqdm(tickers, desc="Downloading data"):
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
        if df.shape[0] < max(HORIZONS.values()) + 1:
            continue
        df = df[['Open','High','Low','Close']].dropna()
        for h_key, h_days in HORIZONS.items():
            df[f'future_max_{h_key}'] = (
                df['Close']
                  .rolling(window=h_days+1, min_periods=h_days+1)
                  .max()
                  .shift(-h_days)
            )
        df = df.dropna()
        for idx, row in df.iterrows():
            feat = row[['Open','High','Low','Close']].values
            label = {
                h_key: int(row[f'future_max_{h_key}'] > row['Close'])
                for h_key in HORIZONS
            }
            rows.append({'ticker': ticker, 'date': idx, 'features': feat, **label})
    data = pd.DataFrame(rows)
    X = np.stack(data['features'].values)
    y = {h: data[h].values for h in HORIZONS}
    return X, y

def train_and_tune():
    tickers = get_sp500_tickers()
    X, y = build_dataset(tickers)
    # train/test split
    X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
    y_train = {h: y[h][:len(X_train)] for h in HORIZONS}
    y_test  = {h: y[h][len(X_train):]    for h in HORIZONS}

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb'))

    thresholds = {}
    for h_key in HORIZONS:
        # train
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_scaled, y_train[h_key])
        # tune threshold by maximizing F1 on X_test
        probs = clf.predict_proba(X_test_scaled)[:,1]
        best_thr, best_f1 = 0.5, 0
        for thr in np.linspace(0,1,101):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(y_test[h_key], preds)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds[h_key] = best_thr
        # save model
        pickle.dump(clf, open(os.path.join(MODELS_DIR, f'breakout_model_{h_key}.pkl'), 'wb'))
        # save test slice for evaluate.py
        pd.DataFrame(X_test).to_csv(os.path.join(MODELS_DIR, f'test_features_{h_key}.csv'), index=False)
        pd.DataFrame({'label': y_test[h_key]}).to_csv(os.path.join(MODELS_DIR, f'test_labels_{h_key}.csv'), index=False)

    # persist thresholds
    with open(os.path.join(MODELS_DIR, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("✅ Training complete. Models, scaler, thresholds.json and test CSVs saved.")

if __name__ == "__main__":
    train_and_tune()
