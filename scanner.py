import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
from concurrent.futures import ThreadPoolExecutor, as_completed

# Replace or extend this with your 100+ ticker list
def get_all_stocks_above_5_dollars():
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "V", "MA",
        "PLTR", "SNOW", "PYPL", "COIN", "F", "GM", "NKE", "WMT", "PEP", "TGT",
        "DIS", "CRM", "ADBE", "ORCL", "AVGO", "UBER", "SQ", "SHOP", "AMD", "INTC"
        # Add more...
    ]

def scan_single_stock(ticker, st_log=None):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty or len(data) < 14:
            if st_log:
                st_log.write(f"⚠️ {ticker}: Not enough price history.")
            return None

        recent_data = data.tail(14)

        # RSI with fallback
        try:
            rsi = RSIIndicator(close=recent_data["Close"]).rsi().iloc[-1]
        except:
            rsi = 50.0

        # Momentum
        try:
            momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0] - 1
        except:
            momentum = 0

        # Volume ratio
        try:
            volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
            volume_change = (recent_data["Volume"].iloc[-1] - recent_data["Volume"].mean()) / recent_data["Volume"].mean() * 100
        except:
            volume_ratio = 1.0
            volume_change = 0

        current_price = round(recent_data["Close"].iloc[-1], 2)

        # Sentiment fallback
        try:
            sentiment_score = get_sentiment_score(ticker)
        except:
            sentiment_score = 0.5

        # Predict breakout score
        try:
            breakout_score = predict_breakout(volume_ratio, momentum + 1, rsi)
        except:
            breakout_score = 0.0

        if st_log:
            st_log.write(f"✅ {ticker} scanned — Breakout: {breakout_score:.3f}")

        return {
            "Ticker": ticker,
            "Current Price": current_price,
            "Breakout Score": round(breakout_score, 3),
            "Target Price": round(current_price * 1.15, 2),
            "Stop Loss": round(current_price * 0.93, 2),
            "RSI": round(rsi, 2),
            "Momentum": round(momentum * 100, 2),
            "Volume Change": round(volume_change, 2),
            "Sentiment Score": round(sentiment_score, 2)
        }

    except Exception as e:
        if st_log:
            st_log.write(f"❌ {ticker} error: {e}")
        return None

def scan_stocks(tickers=None, auto=False, update_progress=None, st_log=None):
    results = []
    tickers = tickers or get_all_stocks_above_5_dollars()
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(scan_single_stock, ticker, st_log): ticker for ticker in tickers}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            if update_progress:
                update_progress((i + 1) / total)

    return results
