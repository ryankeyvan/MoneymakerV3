# scanner.py
import yfinance as yf
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load trained breakout model
MODEL_PATH = "models/breakout_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Tuned F1 threshold from train_model.py output
F1_THRESHOLD = 0.143

# Simple in-memory cache
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
        returns.iloc[-1],                     # yesterday
        returns.iloc[-5:].mean(),             # 5d avg
        returns.iloc[-20:].std(),             # 20d vol
        df["Volume"].iloc[-1] / df["Volume"].iloc[-5:].mean()
    ])

def scan_single_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return {"ticker": ticker, "error": "No data"}
        feat = compute_features(df)
        proba = model.predict_proba([feat])[0][1]
        price = df["Close"].iloc[-1]
        decision = "BUY" if proba >= F1_THRESHOLD else "HOLD"
        return {
            "ticker": ticker,
            "score": round(proba, 4),
            "decision": decision,
            "current_price": round(price, 2),
            "target_price": round(price * 1.10, 2),
            "stop_loss": round(price * 0.95, 2)
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
    ticks = sys.argv[1:]
    print(f"Starting scan of {len(ticks)} tickers… threshold={F1_THRESHOLD}\n")
    out, errs = scan_tickers(ticks)
    print("\n=== RESULTS ===")
    for r in out: print(r)
    if errs:
        print("\n=== ERRORS ===")
        for e in errs: print(e)
