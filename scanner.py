# scanner.py

import yfinance as yf
import pandas as pd
import numpy as np
from ml_model import predict_breakout
import traceback

def get_all_stocks_above_5_dollars():
    return [
        "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "NFLX",
        "AMD", "INTC", "BA", "JPM", "V", "DIS", "UBER", "LYFT", "PYPL", "CRM", "SHOP", "RIVN"
    ]

def scan_single_stock(ticker):
    try:
        df = yf.download(ticker, period="30d", interval="1d", progress=False)

        if df is None or df.empty or len(df) < 14:
            return None, f"{ticker} error: insufficient data"

        df = df.dropna()

        # Make sure we're working with 1D arrays
        close = df['Close'].values.squeeze()
        volume = df['Volume'].values.squeeze()

        if len(close) < 14 or len(volume) < 14:
            return None, f"{ticker} error: not enough clean data"

        # Volume Ratio
        volume_ratio = volume[-1] / np.mean(volume[-14:])

        # Price Momentum
        price_momentum = close[-1] / close[-14]

        # RSI Calculation
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(window=14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

        # ML Prediction
        breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)
        signal = "🔥 Buy" if breakout_score >= 0.7 else "🧐 Watch"

        return {
            "Ticker": ticker,
            "Breakout Score": breakout_score,
            "Volume Ratio": round(volume_ratio, 2),
            "Price Momentum": round(price_momentum, 2),
            "RSI": int(rsi),
            "Signal": signal,
        }, None

    except Exception as e:
        return None, f"{ticker} error: {e}"

def scan_stocks(tickers=None, auto=False, update_progress=lambda p: None):
    if auto or not tickers:
        tickers = get_all_stocks_above_5_dollars()

    results = []
    logs = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        result, log = scan_single_stock(ticker)
        if result:
            results.append(result)
        if log:
            logs.append(f"❌ {log}")
        else:
            logs.append(f"✅ Scanned {ticker}")
        update_progress((i + 1) / total)

    return results, logs
