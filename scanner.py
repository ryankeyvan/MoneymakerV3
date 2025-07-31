# scanner.py

import yfinance as yf
import pandas as pd
from ml_model import predict_breakout

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def scan_stocks(tickers):
    results = []
    logs = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1mo", interval="1d")
            if df.empty or len(df) < 15:
                logs.append(f"⚠️ {ticker} skipped due to insufficient data")
                continue

            df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
            df["momentum"] = df["Close"] / df["Close"].shift(5)
            df["rsi"] = compute_rsi(df["Close"])

            latest = df.iloc[-1]
            prob = predict_breakout(latest["volume_ratio"], latest["momentum"], latest["rsi"])
            signal = "💥 Buy" if prob > 0.75 else "👀 Watch" if prob > 0.3 else "❌ Skip"

            results.append({
                "Ticker": ticker,
                "Breakout Score": prob,
                "Volume Ratio": round(latest["volume_ratio"], 2),
                "Price Momentum": round(latest["momentum"], 2),
                "RSI": round(latest["rsi"], 2),
                "Signal": signal,
            })
        except Exception as e:
            logs.append(f"❌ {ticker} error: {e}")

    return pd.DataFrame(results), logs
