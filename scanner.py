# scanner.py

import yfinance as yf
import pandas as pd
import numpy as np
from ml_model import predict_breakout
import traceback

def get_all_stocks_above_5_dollars():
    # Placeholder list. You can later integrate a live screener here.
    return [
        "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "NFLX",
        "AMD", "INTC", "BA", "JPM", "V", "DIS", "UBER", "LYFT", "PYPL", "CRM", "SHOP", "RIVN"
    ]

def scan_single_stock(ticker):
    try:
        df = yf.download(ticker, period="30d", interval="1d", progress=False)

        if df is None or df.empty or len(df) < 14:
            return None, f"{ticker} error: insufficient data"

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Ensure the price and volume columns exist
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            return None, f"{ticker} error: missing Close or Volume data"

        # Calculate volume ratio (today's volume / 14-day average volume)
        volume_ratio = df['Volume'].iloc[-1] / df['Volume'].tail(14).mean()

        # Calculate price momentum (today's close / close 14 days ago)
        price_momentum = df['Close'].iloc[-1] / df['Close'].iloc[-14]

        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(window=14).mean().iloc[-1]
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Run prediction
        breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

        # Signal generation
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
