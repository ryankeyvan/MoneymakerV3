import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor

def calculate_breakout_score(df):
    if df is None or df.empty:
        return 0
    try:
        rsi = RSIIndicator(df['Close']).rsi().iloc[-1]
        momentum = df['Close'].pct_change(periods=10).iloc[-1] * 100
        volume_change = ((df['Volume'].iloc[-1] - df['Volume'].mean()) / df['Volume'].mean()) * 100
        score = 0
        if 50 < rsi < 70:
            score += 2
        elif rsi >= 70:
            score += 1
        if momentum > 5:
            score += 2
        elif momentum > 2:
            score += 1
        if volume_change > 50:
            score += 2
        elif volume_change > 20:
            score += 1
        return round(score, 2)
    except:
        return 0

def scan_single_stock(ticker):
    try:
        data = yf.download(ticker, period='3mo', interval='1d')
        if data.empty:
            return None
        data.dropna(inplace=True)

        rsi = round(RSIIndicator(data['Close']).rsi().iloc[-1], 2)
        momentum = round(data['Close'].pct_change(10).iloc[-1] * 100, 2)
        volume_change = round(((data['Volume'].iloc[-1] - data['Volume'].mean()) / data['Volume'].mean()) * 100, 2)
        score = calculate_breakout_score(data)
        current_price = round(data['Close'].iloc[-1], 2)

        # Simple projection formula for 1-month price target
        projected_price = round(current_price * (1 + (score / 10)), 2)

        return {
            'Ticker': ticker,
            'Breakout Score': score,
            'RSI': rsi,
            'Momentum': momentum,
            'Volume Change': volume_change,
            'Current Price': current_price,
            'Projected 1M Price': projected_price
        }
    except Exception as e:
        return None

def get_top_100_stocks():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
        "V", "MA", "CRM", "PYPL", "ADBE", "AVGO", "TXN", "QCOM", "ORCL", "IBM",
        "CSCO", "BA", "GE", "CAT", "JNJ", "PFE", "MRK", "ABBV", "TMO", "UNH",
        "WMT", "HD", "COST", "TGT", "LMT", "NOC", "RTX", "CVX", "XOM", "COP",
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "SPGI", "VZ", "T",
        "TMUS", "AMAT", "ASML", "KLAC", "LRCX", "MU", "ENPH", "SEDG", "FSLR", "PLTR",
        "SNOW", "DDOG", "ZS", "PANW", "NET", "CRWD", "DOCU", "ROKU", "SHOP", "SQ",
        "ROST", "TJX", "DG", "DHI", "LEN", "PHM", "NVR", "TSCO", "LOW", "BBY",
        "EA", "TTWO", "ATVI", "MELI", "ETSY", "UBER", "LYFT", "RIVN", "LCID", "F",
        "GM", "DKNG", "DIS", "NKE", "SBUX", "PEP", "KO", "MO", "PM", "CVS"
    ]

def scan_stocks(tickers=None, auto=False):
    if auto:
        tickers = get_top_100_stocks()
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(scan_single_stock, ticker) for ticker in tickers]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return pd.DataFrame(results)
