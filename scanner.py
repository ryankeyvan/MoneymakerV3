import yfinance as yf
from ta.momentum import RSIIndicator
from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout
import pandas as pd
import traceback

def scan_single_stock(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty or len(data) < 14:
            return None, f"⚠️ {ticker}: Not enough data"

        recent = data.tail(14)

        try:
            rsi = RSIIndicator(close=recent["Close"]).rsi().iloc[-1]
        except:
            rsi = 50.0

        try:
            momentum = recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1
        except:
            momentum = 0

        try:
            volume_ratio = recent["Volume"].iloc[-1] / recent["Volume"].mean()
        except:
            volume_ratio = 1.0

        try:
            sentiment = get_sentiment_score(ticker) or "N/A"
        except:
            sentiment = "N/A"

        try:
            breakout_score = predict_breakout(volume_ratio, momentum + 1, rsi)
        except:
            breakout_score = 0.0

        current_price = recent["Close"].iloc[-1]
        target_price = round(current_price * 1.15, 2)
        stop_loss = round(current_price * 0.93, 2)

        return {
            "Ticker": ticker,
            "Breakout Score": round(breakout_score, 3),
            "RSI": round(rsi, 2),
            "Momentum": round(momentum * 100, 2),
            "Volume Change": round((volume_ratio - 1) * 100, 2),
            "Current Price": round(current_price, 2),
            "Target Price": target_price,
            "Stop Loss": stop_loss,
            "Sentiment": sentiment,
            "Signal": "🔥 Buy" if breakout_score >= 0.7 else "🧐 Watch"
        }, f"✅ {ticker} scanned"

    except Exception as e:
        return None, f"❌ {ticker} error: {e}"

def scan_stocks(tickers, update_progress=None):
    import concurrent.futures
    from queue import Queue

    results = []
    logs = []
    q = Queue()
    total = len(tickers)

    def wrapped(ticker):
        result, log = scan_single_stock(ticker)
        if result:
            q.put(result)
        logs.append(log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(wrapped, ticker) for ticker in tickers]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if update_progress:
                update_progress((i + 1) / total)

    while not q.empty():
        results.append(q.get())

    return results, logs

# ✅ Added this missing function
def get_all_stocks_above_5_dollars():
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "V", "MA",
        "JPM", "UNH", "HD", "DIS", "ADBE", "INTC", "AMD", "CRM", "BAC", "KO",
        "PEP", "PFE", "LLY", "ABNB", "AVGO", "T", "XOM", "CVX", "QCOM", "TXN",
        "SBUX", "MRK", "WMT", "COST", "GS", "NKE", "PYPL", "ORCL", "NOW", "CMCSA",
        "GE", "IBM", "MDLZ", "AMAT", "VRTX", "GILD", "TMUS", "REGN", "ISRG", "BKNG",
        "DE", "FDX", "LMT", "RTX", "BLK", "TGT", "MMM", "MO", "CL", "COP",
        "F", "GM", "DAL", "UAL", "CSCO", "ETSY", "PLTR", "SNOW", "SHOP", "ZS",
        "PANW", "DDOG", "DOCU", "ROKU", "TWLO", "BIDU", "UBER", "LYFT", "SQ", "ROST",
        "TJX", "BBY", "EXPE", "MRNA", "BMY", "CVS", "WBA", "CCL", "RCL", "NCLH",
        "DKNG", "CRWD", "NET", "WBD", "PARA", "LULU", "EA", "ATVI", "TTD", "EBAY"
    ]
