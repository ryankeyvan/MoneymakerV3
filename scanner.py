import yfinance as yf
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
import pandas as pd

def run_breakout_scan(ticker_list):
    results = []

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty:
                continue

            recent_data = data.tail(14)
            volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
            price_momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0]
            rsi = 100 - (100 / (1 + (recent_data["Close"].pct_change().mean() / recent_data["Close"].pct_change().std())))

            sentiment = get_sentiment_score(ticker)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            if breakout_score > 0.7:  # Breakout threshold
                results.append({
                    "Ticker": ticker,
                    "Breakout Score": breakout_score,
                    "Target Price": recent_data["Close"].iloc[-1] * 1.15,
                    "Stop Loss": recent_data["Close"].iloc[-1] * 0.93,
                    "Sentiment": sentiment,
                    "Signal": "🔥 Buy"
                })

        except Exception as e:
            print(f"Error with {ticker}: {e}")

    return results
