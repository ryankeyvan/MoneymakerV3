import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

def scan_single_stock(ticker, log_queue=None):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty or len(data) < 14:
            if log_queue:
                log_queue.put(f"⚠️ {ticker}: Not enough price history.")
            return None

        recent_data = data.tail(14)
        rsi = RSIIndicator(close=recent_data["Close"]).rsi().iloc[-1]
        momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0] - 1
        volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
        volume_change = (recent_data["Volume"].iloc[-1] - recent_data["Volume"].mean()) / recent_data["Volume"].mean() * 100
        current_price = round(recent_data["Close"].iloc[-1], 2)
        sentiment_score = get_sentiment_score(ticker)
        breakout_score = predict_breakout(volume_ratio, momentum + 1, rsi)

        if log_queue:
            log_queue.put(f"✅ {ticker} scanned — Breakout: {breakout_score:.3f}")

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
        if log_queue:
            log_queue.put(f"❌ {ticker} error: {e}")
        return None

def scan_stocks(tickers, update_progress=None):
    results = []
    total = len(tickers)
    log_queue = queue.Queue()

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(scan_single_stock, ticker, log_queue): ticker for ticker in tickers}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            if update_progress:
                update_progress((i + 1) / total)

    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())

    return results, logs
