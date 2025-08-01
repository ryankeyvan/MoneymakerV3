#!/usr/bin/env python3
import sys
import os
import json
import pickle

import yfinance as yf
import pandas as pd
from tqdm import tqdm

# — locate this script & your models/ directory —
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

if not os.path.isdir(MODELS_DIR):
    raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

def load_model(fname):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return pickle.load(open(path, 'rb'))

print(f"🔍 Loading models from: {MODELS_DIR}")
print("📂 Available files:", os.listdir(MODELS_DIR))

# — load your three horizons —
models = {
    '1w': load_model('breakout_model_1w.pkl'),
    '1m': load_model('breakout_model_1m.pkl'),
    '3m': load_model('breakout_model_3m.pkl'),
}

# — thresholds.json —
with open(os.path.join(MODELS_DIR, 'thresholds.json'), 'r') as f:
    thresholds = json.load(f)


def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()


def extract_features(df):
    """
    df here already has only ['Open','High','Low','Close'].
    We grab the last row as a 4-element numpy array.
    """
    row = df.iloc[-1]
    return row.values  # shape (4,)


def scan_tickers(tickers):
    results = []
    errors = []

    for ticker in tqdm(tickers, desc=f"Scanning {len(tickers)} tickers…"):
        try:
            df = yf.download(
                ticker,
                period='6mo',
                interval='1d',
                auto_adjust=False,
                progress=False
            )
            if df.empty:
                raise ValueError("No data fetched")

            df = df[['Open', 'High', 'Low', 'Close']]
            current = float(df['Close'].iloc[-1])
            feats   = extract_features(df)

            rec = {
                'ticker':        ticker,
                'current_price': round(current, 2),
                'stop_loss':     round(current * 0.95, 2)
            }

            for h, model in models.items():
                prob     = float(model.predict_proba([feats])[0][1])
                decision = 'BUY' if prob >= thresholds[h] else 'HOLD'

                rec[f'score_{h}']    = round(prob, 4)
                rec[f'decision_{h}'] = decision
                rec[f'target_{h}']   = round(current * (1 + thresholds[h]), 2)

            results.append(rec)

        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})

    return results, errors


def top_breakouts(results, period_key, top_n=5):
    dec   = f"decision_{period_key}"
    score = f"score_{period_key}"
    buys  = [r for r in results if r.get(dec) == 'BUY']
    buys.sort(key=lambda x: x.get(score, 0), reverse=True)
    return buys[:top_n]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        print("⚠️ No tickers provided; defaulting to S&P 500 universe.")
        tickers = get_sp500_tickers()

    results, fetch_errors = scan_tickers(tickers)

    # Print top 5 breakouts for each horizon
    print("\n=== TOP 5 BREAKOUTS ===")

    print("\n📈 1-Week Breakouts:")
    for r in top_breakouts(results, '1w'):
        print(f"  • {r['ticker']}: score={r['score_1w']}, target={r['target_1w']}")

    print("\n📈 1-Month Breakouts:")
    for r in top_breakouts(results, '1m'):
        print(f"  • {r['ticker']}: score={r['score_1m']}, target={r['target_1m']}")

    print("\n📈 3-Month Breakouts:")
    for r in top_breakouts(results, '3m'):
        print(f"  • {r['ticker']}: score={r['score_3m']}, target={r['target_3m']}")

    # Optionally, print any fetch errors
    if fetch_errors:
        print("\n=== ERRORS ===")
        for e in fetch_errors:
            print(f"  {e['ticker']}: {e['error']}")
