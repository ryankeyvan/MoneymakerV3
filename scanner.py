import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

# Dummy ML model to simulate breakout prediction
def predict_breakout(volume_ratio, momentum, rsi):
    features = np.array([[volume_ratio, momentum, rsi]])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Simulate breakout score (normally use trained SVM or MLP)
    score = 0.6 * volume_ratio + 0.3 * momentum + 0.1 * (rsi / 100)
    return round(min(max(score, 0), 1), 2)

def scan_single_stock(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty or "Close" not in data.columns:
            return None

        data.dropna(inplace=True)
        close = data["Close"]
        volume = data["Volume"]

        # RSI Calculation
        rsi = RSIIndicator(close).rsi().iloc[-1]

        # Volume spike
        avg_vol = volume.rolling(window=10).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol != 0 else 0

        # Momentum (% gain over last 10 days)
        momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100 if len(close) >= 10 else 0

        breakout_score = predict_breakout(vol_ratio, momentum, rsi)
        current_price = round(close.iloc[-1], 2)

        return {
            "Ticker": ticker,
            "Breakout Score": breakout_score,
            "Current Price": current_price,
            "RSI": round(rsi, 2),
            "Momentum (%)": round(momentum, 2),
            "Volume Change (%)": round((vol_ratio - 1) * 100, 2),
            "Projected 1M Price": round(current_price * (1 + breakout_score * 0.2), 2),
            "Signal": "🔥 Buy" if breakout_score >= 0.7 else "🧐 Watch"
        }

    except Exception:
        return None

# Get 100 most popular tickers as fallback for auto scan
def get_top_100_stocks():
    return [
        "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "META", "AMZN", "NFLX", "AMD", "INTC",
        "BABA", "CRM", "PYPL", "SHOP", "BA", "DIS", "JPM", "KO", "PEP", "NKE",
        "XOM", "CVX", "PFE", "MRNA", "T", "VZ", "WMT", "COST", "TGT", "MCD",
        "WFC", "GS", "MS", "SBUX", "SQ", "ROKU", "PLTR", "UBER", "LYFT", "SNAP",
        "SOFI", "DKNG", "ABNB", "RBLX", "F", "GM", "TM", "NIO", "RIVN", "LCID",
        "TSM", "QCOM", "TXN", "ADBE", "ORCL", "SAP", "BIDU", "JD", "Z", "ETSY",
        "ZM", "DOCU", "TWLO", "NET", "CRWD", "PANW", "OKTA", "ZS", "DDOG", "MDB",
        "SPOT", "SONY", "TTD", "BB", "GE", "LMT", "RTX", "NOC", "FDX", "UPS",
        "HON", "CAT", "DE", "MMM", "IBM", "CSCO", "HPQ", "MU", "GME", "BBBY",
        "CVS", "UNH", "MRK", "JNJ", "AZN", "LLY", "BIIB", "REGN", "VRTX", "SNY"
    ]

def scan_stocks(tickers=None, auto=False, update_progress=None):
    if auto or not tickers:
        tickers = get_top_100_stocks()

    results = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        result = scan_single_stock(ticker)
        if result:
            results.append(result)

        if update_progress:
            update_progress((idx + 1) / total)

    df = pd.DataFrame(results)
    return df
