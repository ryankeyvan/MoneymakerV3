# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

TICKERS = ["AAPL", "NVDA", "AMD", "MSFT", "TSLA", "GOOGL"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
BREAKOUT_THRESHOLD = 0.1
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def label_breakouts(df, threshold=BREAKOUT_THRESHOLD, window=5):
    df["future_max"] = df["Close"].rolling(window=window, min_periods=1).max().shift(-window)
    df["label"] = (df["future_max"] > df["Close"] * (1 + threshold)).astype(int)
    df.dropna(inplace=True)
    return df

def extract_features(df):
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["momentum"] = df["Close"] / df["Close"].shift(5)
    df["rsi"] = compute_rsi(df["Close"], 14)
    return df[["volume_ratio", "momentum", "rsi", "label"]].dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

all_data = []

for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    if df.empty or "Close" not in df.columns:
        print(f"Skipping {ticker}, no data.")
        continue
    df = label_breakouts(df)
    features = extract_features(df)
    features["ticker"] = ticker
    all_data.append(features)

if not all_data:
    raise ValueError("No data collected for training.")

full_data = pd.concat(all_data)

X = full_data[["volume_ratio", "momentum", "rsi"]]
y = full_data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

print("✅ Training complete.\n")
print(classification_report(y, model.predict(X_scaled)))

joblib.dump(model, os.path.join(MODEL_DIR, "breakout_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("✅ Model and scaler saved to /models/")
