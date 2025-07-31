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
                results.append({
                    'Ticker': ticker,
                    'Breakout Score': 0.0,
                    'Volume Ratio': 'N/A',
                    'Momentum': 'N/A',
                    'RSI': 'N/A',
                    'Sentiment': "No Data",
                    'Target Price': "No Data",
                    'Stop Loss': "No Data",
                    'Buy Signal': False
                })
                continue

            close = data['Close'].values.flatten()
            volume = data['Volume'].values.flatten()
            current_price = float(close[-1])

            price_momentum = close[-1] / np.mean(close[-5:])
            volume_ratio = volume[-1] / np.mean(volume[-20:])
            rsi_series = RSIIndicator(pd.Series(close)).rsi()
            rsi = rsi_series.values.flatten()[-1]

            # Breakout score prediction
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            # Target Price & Stop Loss estimates
            target_price = round(current_price * (1 + breakout_score * 0.15), 2)
            stop_loss = round(current_price * (1 - breakout_score * 0.05), 2)

            # Sentiment is placeholder — real NLP can be added later
            sentiment = (
                "Very Positive" if breakout_score >= 0.8 else
                "Positive" if breakout_score >= 0.65 else
                "Neutral" if breakout_score >= 0.45 else
                "Negative"
            )

            results.append({
                'Ticker': ticker,
                'Breakout Score': float(round(breakout_score, 2)),
                'Volume Ratio': round(volume_ratio, 2),
                'Momentum': round(price_momentum, 2),
                'RSI': round(rsi, 2),
                'Sentiment': sentiment,
                'Target Price': target_price,
                'Stop Loss': stop_loss,
                'Buy Signal': breakout_score >= 0.7
            })

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")
            results.append({
                'Ticker': ticker,
                'Breakout Score': 0.0,
                'Volume Ratio': 'N/A',
                'Momentum': 'N/A',
                'RSI': 'N/A',
                'Sentiment': "Error",
                'Target Price': "Error",
                'Stop Loss': "Error",
                'Buy Signal': False
            })

    return results
