import yfinance as yf
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load trained breakout model
MODEL_PATH = 'models/breakout_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Tuned F1 threshold for breakout classification
F1_THRESHOLD = 0.143

# In-memory cache for historical price data
data_cache = {}


def get_stock_data(ticker, period='6mo', interval='1d'):
    """
    Fetches historical data for a given ticker, with simple caching to reduce API calls.
    """
    cache_key = f"{ticker}_{period}_{interval}"
    if cache_key in data_cache:
        return data_cache[cache_key]
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        data_cache[cache_key] = df
        return df
    except Exception as e:
        raise RuntimeError(f"Data fetch error for {ticker}: {e}")


def compute_features(df):
    """
    Compute feature vector for model prediction. Placeholder for real feature logic.
    """
    # Example features: recent returns, volatility, volume change
    closes = df['Close']
    returns = closes.pct_change().fillna(0)
    feature_vector = [
        returns[-1],                     # yesterday's return
        returns[-5:].mean(),            # 5-day avg return
        returns[-20:].std(),            # 20-day volatility
        df['Volume'][-1] / df['Volume'][-5:].mean()  # volume spike ratio
    ]
    return np.array(feature_vector)


def scan_single_stock(ticker):
    """
    Scans a single ticker for breakout signal, returns dict with details or error.
    """
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return {'ticker': ticker, 'error': 'No data fetched'}

        features = compute_features(df)
        proba = model.predict_proba([features])[0][1]
        current_price = df['Close'].iloc[-1]

        decision = 'BUY' if proba >= F1_THRESHOLD else 'HOLD'
        target_price = current_price * (1 + 0.10)  # +10% projection
        stop_loss = current_price * 0.95           # 5% stop loss

        return {
            'ticker': ticker,
            'score': round(proba, 4),
            'decision': decision,
            'current_price': round(current_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss': round(stop_loss, 2)
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}


def scan_tickers(tickers, max_workers=8):
    """
    Scans a list of tickers in parallel, returns tuple of (results, failures).
    """
    results = []
    failures = []
    # Use ThreadPoolExecutor for parallel scanning
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(scan_single_stock, t): t for t in tickers}
        # Progress bar over completed tasks
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc='Scanning'):
            res = future.result()
            if 'error' in res:
                failures.append(res)
            else:
                results.append(res)

    return results, failures


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scanner.py TICKER1 TICKER2 ...")
        sys.exit(1)

    tickers = sys.argv[1:]
    print(f"Starting scan for {len(tickers)} tickers with threshold={F1_THRESHOLD}\n")
    results, failures = scan_tickers(tickers)

    print("\n=== Scan Results ===")
    for r in results:
        print(r)

    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f)
