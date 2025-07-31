import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
import random
import datetime

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
	
def calculate_sentiment_score(ticker):
    # Placeholder: Random for demo
    return random.uniform(-1, 1)

def scan_stocks(tickers=None, auto=False):
    if auto:
        tickers = get_all_stocks_above_5_dollars()

    if not tickers:
        return []

    results = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="3mo", interval="1d")
            if data.empty or len(data) < 20:
                raise ValueError("Insufficient data")

            data['RSI'] = RSIIndicator(data['Close']).rsi()
            data['Momentum'] = data['Close'].diff()
            data['Volume_Change'] = data['Volume'].pct_change().fillna(0)

            latest = data.iloc[-1]
            rsi = latest['RSI']
            momentum = latest['Momentum']
            volume_change = latest['Volume_Change']
            current_price = latest['Close']

            # Normalize for scoring
            score = 0
            if rsi and not np.isnan(rsi):
                if 50 < rsi < 70:
                    score += 0.3
                elif rsi >= 70:
                    score += 0.1

            if momentum > 0:
                score += 0.2

            if volume_change > 0.25:
                score += 0.2
            elif volume_change > 0.1:
                score += 0.1

            # Sentiment (fake/random placeholder)
            sentiment_score = calculate_sentiment_score(ticker)
            if sentiment_score > 0.3:
                score += 0.2
            elif sentiment_score > 0:
                score += 0.1

            score = round(min(score, 1.0), 2)

            # Optional target price & stop loss estimates
            target_price = round(current_price * (1 + score * 0.15), 2)
            stop_loss = round(current_price * (1 - score * 0.10), 2)

            results.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "RSI": round(rsi, 2),
                "Momentum": round(momentum, 2),
                "Volume Change": round(volume_change * 100, 2),
                "Sentiment Score": round(sentiment_score, 2),
                "Breakout Score": score,
                "Target Price": target_price,
                "Stop Loss": stop_loss
            })

        except Exception as e:
            results.append({
                "Ticker": ticker,
                "Error": str(e)
            })

    return results
