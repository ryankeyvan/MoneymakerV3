import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
import time

def get_all_stocks_above_5_dollars():
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "V", "MA",
        "CRM", "ADBE", "INTC", "AMD", "AVGO", "QCOM", "CSCO", "ORCL", "SHOP", "UBER",
        "PYPL", "SQ", "COIN", "PLTR", "SNOW", "NET", "DDOG", "ZS", "PANW", "CRWD", "DOCU",
        "ROKU", "TWLO", "SPOT", "ABNB", "RBLX", "NKE", "LULU", "DIS", "WMT", "TGT",
        "HD", "LOW", "COST", "SBUX", "MCD", "CMG", "EL", "KO", "PEP", "PM",
        "XOM", "CVX", "COP", "SLB", "OXY", "HAL", "PSX", "MPC", "PXD", "FANG",
        "JPM", "BAC", "WFC", "GS", "MS", "SCHW", "AXP", "C", "USB", "TD",
        "UNH", "JNJ", "PFE", "LLY", "MRK", "BMY", "VRTX", "REGN", "CVS", "CI",
        "TMO", "ISRG", "DHR", "ZBH", "BDX", "GE", "CAT", "DE", "HON", "BA",
        "LMT", "NOC", "RTX", "FDX", "UPS", "DAL", "UAL", "F", "GM", "RIVN"
    ]

def scan_stocks(tickers=None, auto=False):
    results = []
    tickers = tickers or get_all_stocks_above_5_dollars()

    for ticker in tickers:
        try:
            time.sleep(0.3)
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty or len(data) < 14:
                continue

            recent_data = data.tail(14)
            rsi = RSIIndicator(close=recent_data["Close"]).rsi().iloc[-1]
            momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0] - 1
            volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
            volume_change = (recent_data["Volume"].iloc[-1] - recent_data["Volume"].mean()) / recent_data["Volume"].mean() * 100
            current_price = recent_data["Close"].iloc[-1]

            try:
                sentiment_score = get_sentiment_score(ticker)
            except:
                sentiment_score = 0.5  # fallback if Yahoo fails

            breakout_score = predict_breakout(volume_ratio, momentum + 1, rsi)

            results.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Breakout Score": round(breakout_score, 3),
                "Target Price": round(current_price * 1.15, 2),
                "Stop Loss": round(current_price * 0.93, 2),
                "RSI": round(rsi, 2),
                "Momentum": round(momentum * 100, 2),
                "Volume Change": round(volume_change, 2),
                "Sentiment Score": round(sentiment_score, 2)
            })

        except Exception as e:
            print(f"⚠️ Failed on {ticker}: {e}")
            continue

    return results
