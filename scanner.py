# scanner.py (test only)
import yfinance as yf
import numpy as np
import pandas as pd

def scan_stocks(tickers=["AAPL"], auto=False):
    results = []
    for ticker in tickers:
        try:
            print(f"📡 Scanning {ticker}")
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty:
                print(f"⚠️ No data for {ticker}")
                continue

            close_prices = data["Close"]
            volume = data["Volume"]

            price_momentum = close_prices.iloc[-1] / close_prices.iloc[0]
            volume_ratio = volume.iloc[-1] / volume.mean()
            rsi = 50  # fake RSI for test

            breakout_score = (price_momentum + volume_ratio + rsi/100) / 3
            current_price = close_prices.iloc[-1]

            result = {
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Breakout Score": round(breakout_score, 3),
                "Target Price": round(current_price * 1.15, 2),
                "Stop Loss": round(current_price * 0.93, 2),
                "RSI": rsi,
                "Momentum": round((price_momentum - 1) * 100, 2),
                "Volume Change": round((volume.iloc[-1] - volume.mean()) / volume.mean() * 100, 2),
                "Sentiment Score": 0.5
            }

            print(f"✅ {ticker} result:", result)
            results.append(result)

        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")
            continue

    print(f"🔎 Total results: {len(results)}")
    return results
