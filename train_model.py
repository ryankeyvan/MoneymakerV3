import yfinance as yf
import numpy as np
import pandas as pd
import pickle, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Horizon definitions (must match training)
HORIZONS = {
    "1w": {"pct": 0.10},
    "1m": {"pct": 0.15},
    "3m": {"pct": 0.30},
}

# Load trained models
models = {name: pickle.load(open(f"models/breakout_{name}.pkl","rb")) for name in HORIZONS}
# Load thresholds
thresholds = json.load(open("models/thresholds.json","r"))

# Cache for history
data_cache = {}

def get_sp500_tickers():
    """Fetch S&P 500 tickers via Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].tolist()


def get_stock_data(ticker, period="6mo", interval="1d"):
    key = f"{ticker}_{period}_{interval}"
    if key in data_cache:
        return data_cache[key]
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    data_cache[key] = df
    return df


def compute_features(df):
    closes = df['Close']
    returns = closes.pct_change().fillna(0)
    vol = df['Volume']
    return np.array([
        returns.iloc[-1],
        returns.iloc[-5:].mean(),
        returns.iloc[-20:].std(),
        vol.iloc[-1] / vol.iloc[-5:].mean()
    ])


def scan_single_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty or 'Close' not in df:
            return {"ticker":ticker, "error":"No data fetched"}
        price = float(df['Close'].iloc[-1])
        feat = compute_features(df).reshape(1,-1)
        out = {"ticker":ticker, "current_price":round(price,2),
               "stop_loss":round(price*0.95,2)}
        for name, cfg in HORIZONS.items():
            proba = models[name].predict_proba(feat)[0][1]
            thr = thresholds.get(name,0.5)
            out[f"score_{name}"] = round(float(proba),4)
            out[f"decision_{name}"] = "BUY" if proba>=thr else "HOLD"
            out[f"target_{name}"] = round(price*(1+cfg['pct']),2)
        return out
    except Exception as e:
        return {"ticker":ticker, "error":str(e)}


def scan_tickers(tickers=None, max_workers=8):
    if not tickers:
        tickers = get_sp500_tickers()
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_single_stock, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            res = fut.result()
            (failures if 'error' in res else results).append(res)
    return results, failures


if __name__ == '__main__':
    import sys
    ticks = sys.argv[1:] if len(sys.argv)>1 else []
    if not ticks:
        print("⚠️ No tickers provided; defaulting to S&P 500 universe.")
    print(f"🔎 Scanning {len(ticks) or '500+'} tickers...")
    res, errs = scan_tickers(ticks)
    print("\n=== BREAKOUT COLORS ===")
    for r in res:
        print(r)
    if errs:
        print("\n=== ERRORS ===")
        for e in errs:
            print(e)
