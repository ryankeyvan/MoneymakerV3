# scanner.py

import yfinance as yf
from datetime import datetime, timedelta
from ml_model import predict_breakout
import numpy as np

def get_all_stocks_above_5_dollars():
    # You can customize this list with a screener in the future
    return ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "AMD", "INTC"]

def scan_stocks(tickers, update_progress=lambda x: None):
    results = []
    logs = []

    for i, ticker in enumerate(tickers):
        update_progress(i / len(tickers))
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            df = stock.history(start=start_date, end=end_date)

            if df.empty or len(df) < 14:
                logs.append(f"⚠️ {ticker} error: Not enough data.")
                continue

            # Calculate features
            recent_close = df["Close"].iloc[-1]
            avg_close = df["Close"].mean()
            price_momentum = round(recent_close / avg_close, 2)

            recent_volume = df["Volume"].iloc[-1]
            avg_volume = df["Volume"].mean()
            volume_ratio = round(recent_volume / avg_volume, 2)

            rsi = compute_rsi(df["Close"].values, window=14)

            if rsi is None:
                logs.append(f"⚠️ {ticker} error: RSI calculation failed.")
                continue

            # Predict using ML model (make sure it's single feature vector)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            # Signal logic
            signal = "🔥 Buy" if breakout_score >= 0.7 else "🧐 Watch"

            results.append({
                "Ticker": ticker,
                "Breakout Score": breakout_score,
                "RSI": round(rsi, 2),
                "Momentum": price_momentum,
                "Volume Change": volume_ratio,
                "Signal": signal
            })

        except Exception as e:
            logs.append(f"❌ {ticker} error: {str(e)}")

    update_progress(1.0)
    return results, logs

def compute_rsi(prices, window=14):
    try:
        prices = np.array(prices)
        if len(prices) < window:
            return None
        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed > 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return None
