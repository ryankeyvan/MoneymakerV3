import yfinance as yf
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from utils.preprocessing import preprocess_for_training

# === Config ===
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
FUTURE_DAYS = 5
BREAKOUT_THRESHOLD = 1.10  # 10% breakout target

X_all = []
y_all = []

print("📊 Starting training process...")

for ticker in TICKERS:
    print(f"\n📈 Fetching {ticker}...")

    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)

        if df.empty or "Close" not in df.columns
