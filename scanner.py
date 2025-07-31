import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
import numpy as np

def calculate_indicators(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['Momentum'] = df['Close'].pct_change(periods=14) * 100
    df['Volume_Change'] = df['Volume'].pct_change(periods=3) * 100
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    return df

def score_breakout(df):
    latest = df.iloc[-1]
    score = 0
    if 50 < latest['RSI'] < 70:
        score += 1
    if latest['Momentum'] > 0:
        score += 1
    if latest['Volume_Change'] > 0:
        score += 1
    if latest['OBV'] > df['OBV'].mean():
        score += 1
    return round(score * 0.25, 2)  # 0–1 scale

def project_1m_price(df):
    recent_growth = df['Close'].pct_change().rolling(window=5).mean().iloc[-1]
    projected = df['Close'].iloc[-1] * (1 + (recent_growth * 21))  # ~21 trading days
    return round(projected, 2)

def scan_stocks(tickers, auto=False, update_progress=None, st_log=None):
    results = []
    logs = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty or len(data) < 20:
                msg = f"⚠️ No data for {ticker}"
                if st_log: st_log.write(msg)
                logs.append(msg)
                continue

            df = calculate_indicators(data.dropna())
            if df.empty:
                msg = f"⚠️ Insufficient data after indicator calc for {ticker}"
                if st_log: st_log.write(msg)
                logs.append(msg)
                continue

            score = score_breakout(df)
            current_price = round(df['Close'].iloc[-1], 2)
            target_price = project_1m_price(df)
            stop_loss = round(current_price * 0.93, 2)

            result = {
                "Ticker": ticker,
                "Current Price": current_price,
                "Breakout Score": score,
                "Projected 1M Price": target_price,
                "Stop Loss": stop_loss,
                "Signal": "🔥 Buy" if score >= 0.7 else "🧐 Watch"
            }
            results.append(result)

        except Exception as e:
            msg = f"❌ {ticker} error: {e}"
            if st_log: st_log.write(msg)
            logs.append(msg)

        if update_progress:
            update_progress((i + 1) / total)

    return results, logs

def get_all_stocks_above_5_dollars():
    # Placeholder — replace with your source of 100+ tickers over $5
    return ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AMD", "INTC", "BA", "JPM", "V", "DIS", "UBER", "LYFT", "PLTR", "SOFI", "SNAP", "SHOP", "COIN"]
