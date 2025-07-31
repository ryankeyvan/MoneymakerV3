import yfinance as yf
import numpy as np
from ta.momentum import RSIIndicator
from ml_model import predict_breakout
from utils.sentiment import get_sentiment_score

def scan_stocks(ticker_list):
    results = []

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty or len(data) < 14:
                results.append({
                    "Ticker": ticker,
                    "Breakout Score": 0,
                    "Volume Ratio": "N/A",
                    "Momentum": "N/A",
                    "RSI": "N/A",
                    "Sentiment": "N/A",
                    "Current Price": "N/A",
                    "Target Price": "N/A",
                    "Stop Loss": "N/A",
                    "Buy Signal": False
                })
                continue

            recent = data.tail(14)
            volume_ratio = recent["Volume"].iloc[-1] / recent["Volume"].mean()
            momentum = recent["Close"].iloc[-1] / recent["Close"].iloc[0]
            rsi_calc = RSIIndicator(recent["Close"])
            rsi = rsi_calc.rsi().iloc[-1]
            current_price = recent["Close"].iloc[-1]

            sentiment = get_sentiment_score(ticker)
            score = predict_breakout(volume_ratio, momentum, rsi)

            result = {
                "Ticker": ticker,
                "Breakout Score": round(score, 3),
                "Volume Ratio": round(volume_ratio, 2),
                "Momentum": round(momentum, 3),
                "RSI": round(rsi, 2),
                "Sentiment": sentiment,
                "Current Price": current_price,
                "Target Price": current_price * 1.15,
                "Stop Loss": current_price * 0.93,
                "Buy Signal": score >= 0.7
            }

            results.append(result)

        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Breakout Score": 0,
                "Volume Ratio": "N/A",
                "Momentum": "N/A",
                "RSI": "N/A",
                "Sentiment": f"Error: {e}",
                "Current Price": "N/A",
                "Target Price": "N/A",
                "Stop Loss": "N/A",
                "Buy Signal": False
            })

    return results
