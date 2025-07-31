import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
import concurrent.futures

def scan_single_stock(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 14:
            return None, f"⚠️ {ticker} has insufficient data."

        recent = df.tail(14)
        current_price = df["Close"].iloc[-1]
        volume_change = ((recent["Volume"].iloc[-1] - recent["Volume"].mean()) / recent["Volume"].mean()) * 100
        momentum = ((recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent["Close"].iloc[0]) * 100
        rsi = RSIIndicator(close=recent["Close"]).rsi().iloc[-1]
        sentiment = get_sentiment_score(ticker)
        breakout_score = predict_breakout(volume_change, momentum, rsi)

        signal = "🔥 Buy" if breakout_score >= 0.7 else "🧐 Watch"

        result = {
            "Ticker": ticker,
            "Current Price": round(current_price, 2),
            "Breakout Score": round(breakout_score, 3),
            "RSI": round(rsi, 1),
            "Momentum (%)": round(momentum, 2),
            "Volume Change (%)": round(volume_change, 2),
            "Sentiment": sentiment,
            "Target Price (1M)": round(current_price * 1.15, 2),
            "Stop Loss": round(current_price * 0.93, 2),
            "Signal": signal,
        }

        return result, f"✅ {ticker} scanned."
    except Exception as e:
        return None, f"❌ {ticker} error: {e}"

def scan_stocks(tickers, auto=False, update_progress=None):
    results = []
    logs = []
    total = len(tickers)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(scan_single_stock, ticker): ticker for ticker in tickers}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result, log = future.result()
            if result:
                results.append(result)
            logs.append(log)
            if update_progress:
                update_progress(i / total)

    return results, logs

def get_all_stocks_above_5_dollars():
    # Replace with real dynamic scanner later
    return [
        "AAPL", "MSFT", "TSLA", "NVDA", "META", "GOOGL", "AMZN", "NFLX", "AMD", "SHOP",
        "BA", "CRM", "INTC", "V", "MA", "PYPL", "BABA", "UBER", "DIS", "PLTR",
        "SNAP", "RIVN", "NIO", "DKNG", "ABNB", "COIN", "T", "WMT", "SBUX", "COST",
        "ORCL", "SQ", "QCOM", "F", "GM", "CVX", "XOM", "PEP", "KO", "MCD",
        "PFE", "MRNA", "JNJ", "UNH", "LLY", "VRTX", "BMY", "GILD", "TMO", "ISRG",
        "ZS", "CRWD", "PANW", "DDOG", "NET", "MDB", "TWLO", "OKTA", "ADBE", "NOW",
        "VEEV", "ROKU", "TTD", "ETSY", "DOCU", "ZI", "ZM", "BILL", "FSLY", "AFRM",
        "NEE", "DUK", "SO", "EXC", "D", "AEP", "ED", "PEG", "EIX", "SRE",
        "AGX", "CREV", "LLHAI", "LIDR", "OCTO", "HTOO", "DASH", "V", "BX", "GS"
    ]
