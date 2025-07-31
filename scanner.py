# scanner.py

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from ta.momentum import RSIIndicator
from ml_model import predict_breakout

def scan_stocks(tickers):
    results = []
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=90)

    for ticker in tickers:
        try:
            print(f"🔎 Scanning {ticker}...")
            data = yf.download(ticker, start=start, end=end, progress=False)

            if data.empty or len(data) < 20:
                print(f"⚠️ No data for {ticker}")
                continue

            close = data['Close'].values.flatten()
            volume = data['Volume'].values.flatten()

            # Price momentum (today's close / 5-day average)
            price_momentum = close[-1] / np.mean(close[-5:])

            # Volume ratio (today's volume / 20-day avg volume)
            volume_ratio = volume[-1] / np.mean(volume[-20:])

            # RSI calculation
            rsi_series = RSIIndicator(pd.Series(close)).rsi()
            rsi = rsi_series.values.flatten()[-1]

            # Predict breakout score (0 to 1)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            # Build result
            results.append({
                'Ticker': ticker,
                'Breakout Score': float(breakout_score),
                'Volume Ratio': round(volume_ratio, 2),
                'Momentum': round(price_momentum, 2),
                'RSI': round(rsi, 2),
                'Sentiment': 'N/A',  # Optional bonus: integrate Twitter/X later
                'Target Price': 'N/A',  # Optional: could add ML model here
                'Stop Loss': 'N/A',  # Optional: ATR-based stop loss later
                'Buy Signal': breakout_score >= 0.7
            })

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
            continue

    return results
