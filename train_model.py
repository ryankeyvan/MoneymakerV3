# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Parameters
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
BREAKOUT_THRESHOLD = 0.10  # 10% gain in next N days
FUTURE_DAYS = 5

all_data = []

def label_breakouts(df, threshold=BREAKOUT_THRESHOLD, future_days=FUTURE_DAYS):
    # Ensure we have required column
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column")

    df = df.copy()
    df["future_max"] = df["Close"].shift(-1).rolling(window=future_days).max()
    df.dropna(subset=["future_max"], inplace=True)

    df["label"] = (df["future_max"] > df["Close"] * (1 + threshold)).astype(int)
    return df

def calculate_indicators(df):
    df = df.copy()
    df["momentum"] = df["Close"] - df["Close"].shift(5)
    df["rsi"] = compute_rsi(df["Close"], window=14)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window=5).mean()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        if df.empty:
            raise ValueError("DataFrame is empty.")

        df = label_breakouts(df)
        df = calculate_indicators(df)
        df["ticker"] = ticker
        all_data.append(df[["momentum", "rsi", "volume_ratio", "label", "ticker"]])
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

# Com
