import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_all_stocks_above_5_dollars():
    return [
        # (100+ tickers omitted here for brevity — use your full list from before)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "V", "MA",
        "PLTR", "SNOW", "PYPL", "COIN", "F", "GM", "NKE", "WMT", "PEP", "TGT",
        "DIS", "CRM", "ADBE", "ORCL", "AVGO", "UBER", "SQ", "SHOP", "AMD", "INTC"
        # ... add full list here
    ]

def scan_single_stock(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty or len(data) < 14:
            return None

        recent_data = data.tail(14)
        rsi = RSIIndicator(close=recent_data["Close"]).rsi().iloc[-1]
        momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0] - 1
        volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
        volume_change = (recent_data["Volume"].iloc[-1] - recent_data["Volume"].mean()) / recent_data["Volume"].mean() * 100
        current_price = recent_data["Close"].iloc[-1]

        try:
            sentiment_score = get_sentiment_score(ticker)
        except:
            sentiment_score = 0.5

        breakout_score = predict_breakout(volume_ratio, momentum + 1, rsi)

        return {
            "Ticker": ticker,
            "Current Price": round(current_price, 2),
            "Breakout Score": round(breakout_score, 3),
            "Target Price": round(current_price * 1.15, 2),
            "Stop Loss": round(current_price * 0.93, 2),
            "RSI": round(rsi, 2),
            "Momentum": round(momentum * 100, 2),
            "Volume Change": round(volume_change, 2),
            "Sentiment Score": round(sentiment_score, 2)
        }
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")
        return None

def scan_stocks(tickers=None, auto=False, update_progress=None):
    results = []
    tickers = tickers or get_all_stocks_above_5_dollars()
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(scan_single_stock, ticker): ticker for ticker in tickers}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            if update_progress:
                update_progress((i + 1) / total)

    return results
