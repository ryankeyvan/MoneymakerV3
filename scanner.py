# scanner.py

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ml_model import predict_breakout

def get_volume_ratio(df):
    return df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]

def get_momentum(df):
    return df["Close"].iloc[-1] / df["Close"].iloc[-2]

def get_rsi(df):
    return RSIIndicator(close=df["Close"]).rsi().iloc[-1]

def scan_stocks(tickers):
    results = []
    logs = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="30d", interval="1d", progress=False)
            if df.shape[0] < 20:
                logs.append(f"⚠️ {ticker} skipped: Not enough data")
                continue

            volume_ratio = get_volume_ratio(df)
            momentum = get_momentum(df)
            rsi = get_rsi(df)

            score = predict_breakout(volume_ratio, momentum, rsi)

            signal = "🧐 Watch"
            if isinstance(score, float):
                if score > 0.8:
                    signal = "🚀 Strong Buy"
                elif score > 0.6:
                    signal = "🟢 Buy"
                elif score < 0.3:
                    signal = "🔴 Avoid"

            results.append({
                "Ticker": ticker,
                "Breakout Score": score,
                "Volume Ratio": round(volume_ratio, 2),
                "Price Momentum": round(momentum, 2),
                "RSI": round(rsi, 2),
                "Signal": signal
            })

        except Exception as e:
            logs.append(f"❌ {ticker} error: {str(e)}")

    return pd.DataFrame(results), logs
