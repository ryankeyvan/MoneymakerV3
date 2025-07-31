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

# Tuned F1 threshold from train_model.py output
F1_THRESHOLD = 0.180

# Simple in-memory cache for historical data
data_cache = {}


def get_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetches historical data for a given ticker, with caching to reduce API calls.
    """
    key = f"{ticker}_{period}_{interval}"
    if key in data_cache:
        return data_cache[key]
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    data_cache[key] = df
    return df


def compute_features(df):
    """
    Compute feature vector for model prediction.
    """
    closes = df["Close"]
    returns = closes.pct_change().fillna(0)
    return np.array([
        returns.iloc[-1],                    # last-day return
        returns.iloc[-5:].mean(),            # 5-day avg return
        returns.iloc[-20:].std(),            # 20-day volatility
        df["Volume"].iloc[-1] / df["Volume"].iloc[-5:].mean()  # volume spike ratio
    ])


def scan_single_stock(ticker):
    """
    Scans a single ticker for breakout signal.
    """
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return {"ticker": ticker, "error": "No data fetched"}

        # Handle multi-index columns (if yfinance downloaded multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                df = df[ticker]
            else:
                return {"ticker": ticker, "error": f"{ticker} not found in downloaded data"}

        # Extract last closing price as float
        last_price = float(df["Close"].iloc[-1])

        # Compute features and predict probability
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
    """
    Scans a list of tickers in parallel.
    """
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
