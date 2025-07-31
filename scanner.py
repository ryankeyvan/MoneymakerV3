import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
from model import predict_breakout
import datetime
import numpy as np
import concurrent.futures

# Time range for historical data
END_DATE = datetime.datetime.now()
START_DATE = END_DATE - datetime.timedelta(days=90)

# Scan a single stock
def scan_single_stock(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty or 'Close' not in df.columns:
            raise ValueError("No price data")

        df.dropna(inplace=True)
        close = df['Close'].values  # Get numpy array
        volume = df['Volume'].values

        if len(close) < 15 or len(volume) < 15:
            raise ValueError("Not enough data")

        rsi = RSIIndicator(close=pd.Series(close.flatten())).rsi().values[-1]
        macd = MACD(close=pd.Series(close.flatten())).macd_diff().values[-1]
        obv = OnBalanceVolumeIndicator(close=pd.Series(close.flatten()), volume=pd.Series(volume.flatten())).on_balance_volume().values[-1]

        momentum = ((close[-1] - close[-15]) / close[-15]) * 100
        volume_change = ((volume[-1] - np.mean(volume[-15:])) / np.mean(volume[-15:])) * 100

        features = pd.DataFrame([{
            "rsi": rsi,
            "macd": macd,
            "obv": obv,
            "momentum": momentum,
            "volume_change": volume_change
        }])

        scaled = MinMaxScaler().fit_transform(features)
        score = predict_breakout(scaled)

        return {
            "Ticker": ticker,
            "Breakout Score": round(float(score), 3),
            "RSI": round(rsi, 2),
            "Momentum": round(momentum, 2),
            "Volume Change": round(volume_change, 2),
            "Current Price": round(float(close[-1]), 2),
            "Signal": "🔥 Buy" if score >= 0.7 else "🧐 Watch"
        }

    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# Multi-threaded scanner
def scan_stocks(tickers, update_progress=None):
    results = []
    logs = []
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(scan_single_stock, ticker) for ticker in tickers]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if "Error" in result:
                logs.append(f"❌ {result['Ticker']} error: {result['Error']}")
            else:
                results.append(result)
            if update_progress:
                update_progress((i + 1) / total)

    return results, logs

# Load a list of popular stocks (>$5)
def get_all_stocks_above_5_dollars():
    return [
        "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA", "NFLX",
        "INTC", "AMD", "BA", "JPM", "V", "MA", "DIS", "UBER", "LYFT", "PLTR",
        "SQ", "PYPL", "CRM", "SHOP", "ABNB", "COIN", "ROKU", "NKE", "TGT",
        "COST", "WMT", "PFE", "MRNA", "BNTX", "XOM", "CVX", "F", "GM", "RIVN",
        "LCID", "UAL", "DAL", "AAL", "SBUX", "QCOM", "ZM", "DDOG", "SNOW", "NET",
        "TWLO", "DOCU", "SOFI", "ENPH", "SPWR", "FSLR"
    ]
