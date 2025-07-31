import yfinance as yf
import numpy as np
import pandas as pd
from ml_model import predict_breakout

def scan_stocks(tickers):
    results = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty:
                results.append({"Ticker": ticker, "Error": "No data"})
                continue

            volume = data['Volume'].values
            close = data['Close'].values
            if len(volume) < 20 or len(close) < 20:
                results.append({"Ticker": ticker, "Error": "Insufficient data"})
                continue

            # Feature engineering
            avg_vol = np.mean(volume[-20:])
            volume_ratio = volume[-1] / avg_vol if avg_vol != 0 else 0
            price_momentum = close[-1] / close[-5] if close[-5] != 0 else 0

            delta = pd.Series(close).diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(window=14).mean()
            avg_loss = pd.Series(loss).rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50

            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            results.append({
                "Ticker": ticker,
                "Breakout Score": float(breakout_score),
                "Target Price": None,  # Optional: you can add calc here
                "Stop Loss": None,
                "Sentiment": "very positive" if breakout_score > 0.7 else "neutral"
            })

        except Exception as e:
            results.append({"Ticker": ticker, "Error": str(e)})

    return results
