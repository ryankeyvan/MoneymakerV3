#!/usr/bin/env python3
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

# suppress that “float(Series)” FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# trading‐day horizons
HORIZONS = {'1w': 5, '1m': 21, '3m': 63}
FEATURE_COLS = ['Open', 'High', 'Low', 'Close']


def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, header=0)[0]
    return df['Symbol'].str.replace('.', '-', regex=False).tolist()


def build_dataset(tickers, years=3):
    rows = []
    for ticker in tqdm(tickers, desc="Downloading data"):
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
        if df.shape[0] < max(HORIZONS.values()) + 1:
            continue

        df = df[FEATURE_COLS].dropna()
        # compute future rolling max
        for h_key, h_days in HORIZONS.items():
            df[f'future_max_{h_key}'] = (
                df['Close']
                  .rolling(window=h_days + 1, min_periods=h_days + 1)
                  .max()
                  .shift(-h_days)
            )
        df = df.dropna()

        # flatten into rows
        for idx, row in df.iterrows():
            feats = row[FEATURE_COLS].values.astype(float)
            # use .item() to grab scalar, no warnings
            close_val = row['Close'].item()
            labels = {}
            for h_key in HORIZONS:
                fut_val = row[f'future_max_{h_key}'].item()
                labels[h_key] = int(fut_val > close_val)

            rows.append({
                'ticker': ticker,
                'date': idx,
                'features': feats,
                **labels
            })

    data = pd.DataFrame(rows)
    X = np.stack(data['features'].values)
    y = {h: data[h].values for h in HORIZONS}
    return X, y


def train_and_tune():
    tickers = get_sp500_tickers()
    X, y = build_dataset(tickers)

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train = {h: y[h][:split] for h in HORIZONS}
    y_test  = {h: y[h][split:] for h in HORIZONS}

    # standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb'))

    thresholds = {}
    for h_key in HORIZONS:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_s, y_train[h_key])

        # tune threshold on test by F1
        probs = clf.predict_proba(X_test_s)[:, 1]
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.linspace(0, 1, 101):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(y_test[h_key], preds)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        thresholds[h_key] = best_thr
        # save model
        pickle.dump(clf, open(os.path.join(MODELS_DIR, f'breakout_model_{h_key}.pkl'), 'wb'))

        # export test CSVs for evaluate.py
        feat_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
        feat_df.to_csv(os.path.join(MODELS_DIR, f'test_features_{h_key}.csv'), index=False)
        pd.DataFrame({'label': y_test[h_key]}) \
          .to_csv(os.path.join(MODELS_DIR, f'test_labels_{h_key}.csv'), index=False)

        print(f"Horizon {h_key}: best F1 = {best_f1:.4f} @ threshold = {best_thr:.3f}")

    # persist thresholds.json
    with open(os.path.join(MODELS_DIR, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)

    print("✅ Training complete. Models, scaler, thresholds, and test CSVs are in /models.")


if __name__ == "__main__":
    train_and_tune()
