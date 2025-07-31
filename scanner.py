import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load trained breakout model
MODEL_PATH = "models/breakout_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Tuned F1 threshold
F1_THRESHOLD = 0.180

# In-memory cache for historical data
data_cache = {}


def get_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch price history via yfinance.Ticker to avoid multi-index issues.
    """
    key = f"{ticker}_{period}_{interval}"
    if key in data_cache:
        return data_cache[key]
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    data_cache[key] = df
    return df


def compute_features(df):
    """Compute the 4 breakout features from price & volume."""
    closes = df["Close"]
    returns = closes.pct_change().fillna(0)
    return np.array([
        returns.iloc[-1],                 # last-day return
        returns.iloc[-5:].mean(),         # 5-day avg return
        returns.iloc[-20:].std(),         # 20-day volatility
        df["Volume"].iloc[-1] / df["Volume"].iloc[-5:].mean()  # volume spike
    ])


def scan_single_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return {"ticker": ticker, "error": "No data fetched"}
        if "Close" not in df.columns or "Volume" not in df.columns:
            return {"ticker": ticker, "error": "Unexpected data format"}

        last_price = float(df["Close"].iloc[-1])
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
    """Parallel scan of multiple tickers."""
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_single_stock, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Scanning"):
            res = future.result()
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
