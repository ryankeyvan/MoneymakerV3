#!/usr/bin/env python3
import sys
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import pickle
import json

# Load trained models
models = {
    '1w': pickle.load(open('models/breakout_model_1w.pkl', 'rb')),
    '1m': pickle.load(open('models/breakout_model_1m.pkl', 'rb')),
    '3m': pickle.load(open('models/breakout_model_3m.pkl', 'rb'))
}

# Load saved thresholds for each horizon
# thresholds.json should look like: { "1w": 0.18, "1m": 0.26, "3m": 0.21 }
with open('models/thresholds.json', 'r') as f:
    thresholds = json.load(f)


def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia and remaps tickers
    like 'BRK.B' -> 'BRK-B' for Yahoo Finance compatibility.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    tickers = table['Symbol'].tolist()
    # remap any tickers containing a dot to use a hyphen
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers


def extract_features(df):
    """
    Recreate the feature-extraction logic from train_model.py.
    Here we just grab the latest OHLCV row, but you can expand
    this to include TA indicators etc.
    """
    row = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1]
    return row.values


def calculate_target_price(current_price, pct):
    """
    Calculates a simple target price based on a breakout percentage.
    """
    return round(current_price * (1 + pct), 2)


def get_stop_loss(current_price, pct=0.05):
    """
    Sets a stop loss at a fixed percentage below current price.
    """
    return round(current_price * (1 - pct), 2)


def scan_tickers(tickers):
    """
    Scans each ticker, applies all three models, and returns only
    those flagged BUY on the 1-month horizon, plus any fetch errors.
    """
    results = []
    errors = []
    for ticker in tqdm(tickers, desc=f"Scanning {len(tickers)} tickers…"):
        try:
            df = yf.download(ticker, period='6mo', interval='1d', progress=False)
            if df.empty:
                raise ValueError("No data fetched")
            current = df['Close'].iloc[-1]
            feats = extract_features(df)

            record = {
                'ticker': ticker,
                'current_price': round(current, 2),
                'stop_loss': get_stop_loss(current)
            }
            # run each horizon
            for h, model in models.items():
                prob = model.predict_proba([feats])[0][1]
                decision = 'BUY' if prob >= thresholds[h] else 'HOLD'
                record[f'score_{h}'] = round(prob, 4)
                record[f'decision_{h}'] = decision
                record[f'target_{h}'] = calculate_target_price(current, thresholds[h])
            results.append(record)
        except Exception as e:
            errors.append({ 'ticker': ticker, 'error': str(e) })
    # filter to only those the model wants you to BUY in 1-month horizon
    buys = [r for r in results if r['decision_1m'] == 'BUY']
    return buys, errors


if __name__ == '__main__':
    # parse tickers from command line, or default to S&P 500
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        print("⚠️ No tickers provided; defaulting to S&P 500 universe.")
        tickers = get_sp500_tickers()

    buy_list, fetch_errors = scan_tickers(tickers)

    # output only the breakout candidates
    print("=== BREAKOUT CANDIDATES (1m) ===")
    for r in buy_list:
        print(r)

    if fetch_errors:
        print("\n=== ERRORS ===")
        for e in fetch_errors:
            print(e)
