# scanner.py
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MODEL_PATH = "models/breakout_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Use the threshold your training script just printed
F1_THRESHOLD = 0.180

# Simple in-memory cache for historical data
data_cache = {}

def get_stock_data(ticker, period="6mo", interval="1d"):
    key = f"{ticker}_{period}_{interval}"
    if key in data_cache:
        return data_cache[key]
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    data_cache[key] = df
    return df

def compute_features(df):
    closes = df["Close"]
    returns = closes.pct_change().fillna(0)
    return np.array([
        returns.iloc[-1],                   # last-day return
        returns.iloc[-5:].mean(),           # 5-day avg return
        returns.iloc[-20:].std(),           # 20-day volatility
        df["Volume"].iloc[-1] / df["Volume"].iloc[-5:].mean()  # volume spike
    ])

def scan_single_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return {"ticker": ticker, "error": "No data fetched"}

        # Ensure we extract a single float, not a Series
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes[ticker]       # when yf.download returns multi-column
        last_price = float(closes.iloc[-1])

        feat = compute_features(df).reshape(1, -1)
        proba = model.predict_proba(feat)[0][1]

        return {
            "ticker": ticker,
            "score": round(float(proba), 4),
            "decision": "BUY" if proba >= F1_THRESHOLD else "HOLD",
            "current_price": round(last_price, 2),
            "target_price": round(last_price * 1.10, 2),
            "stop_loss": round(last_price * 0.95, 2)
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def scan_tickers(tickers, max_workers=8):
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_single_stock, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(tickers), desc="Scanning"):
            res = fut.result()
            (failures if "error" in res else results).append(res)
    return results, failures

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scanner.py TICKER1 TICKER2 …")
        sys.exit(1)

    tickers = sys.argv[1:]
    print(f"Starting scan of {len(tickers)} tickers… threshold={F1_THRESHOLD}\n")
    results, errors = scan_tickers(tickers)

    print("\n=== RESULTS ===")
    for r in results:
        print(r)
    if errors:
        print("\n=== ERRORS ===")
        for e in errors:
            print(e)
