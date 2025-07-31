#!/usr/bin/env python3
import sys
import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import pickle
import json

# ——— locate this script’s folder and the models/ subfolder ———
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# ——— sanity check ———
if not os.path.isdir(MODELS_DIR):
    raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

print(f"🔍 Loading models from: {MODELS_DIR}")
print("📂 Available files:", os.listdir(MODELS_DIR))

def load_model(fname):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected model file {fname} not found in {MODELS_DIR}."
        )
    return pickle.load(open(path, 'rb'))

# ——— load your 3 horizons ———
models = {
    '1w': load_model('breakout_1w.pkl'),
    '1m': load_model('breakout_1m.pkl'),
    '3m': load_model('breakout_3m.pkl'),
}

# ——— thresholds.json ———
with open(os.path.join(MODELS_DIR, 'thresholds.json'), 'r') as f:
    thresholds = json.load(f)


def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()


def extract_features(df):
    # only keep OHLC → exactly 4 features
    row = df[['Open', 'High', 'Low', 'Close']].iloc[-1]
    return row.values


def calculate_target_price(current_price, pct):
    return round(current_price * (1 + pct), 2)


def get_stop_loss(current_price, pct=0.05):
    return round(current_price * (1 - pct), 2)


def scan_tickers(tickers):
    results, errors = [], []
    for ticker in tqdm(tickers, desc=f"Scanning {len(tickers)} tickers…"):
        try:
            df = yf.download(
                ticker,
                period='6mo',
                interval='1d',
                auto_adjust=False,   # ← preserve raw OHLC
                progress=False
            )
            if df.empty:
                raise ValueError("No data fetched")

            # strip out everything but OHLC
            df = df[['Open', 'High', 'Low', 'Close']]

            current = df['Close'].iloc[-1]
            feats   = extract_features(df)

            rec = {
                'ticker': ticker,
                'current_price': round(current, 2),
                'stop_loss': get_stop_loss(current)
            }
            for h, model in models.items():
                prob     = model.predict_proba([feats])[0][1]
                decision = 'BUY' if prob >= thresholds[h] else 'HOLD'
                rec[f'score_{h}']    = round(prob, 4)
                rec[f'decision_{h}'] = decision
                rec[f'target_{h}']   = calculate_target_price(current, thresholds[h])
            results.append(rec)

        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})

    buys = [r for r in results if r['decision_1m'] == 'BUY']
    return buys, errors


if __name__ == '__main__':
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        print("⚠️ No tickers provided; defaulting to S&P 500 universe.")
        tickers = get_sp500_tickers()

    buy_list, fetch_errors = scan_tickers(tickers)

    print("\n=== BREAKOUT CANDIDATES (1m) ===")
    for r in buy_list:
        print(r)

    if fetch_errors:
        print("\n=== ERRORS ===")
        for e in fetch_errors:
            print(e)
