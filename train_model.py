import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os

# -------------------------
# Config
# -------------------------
TICKER = "AAPL"
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
BREAKOUT_THRESHOLD = 0.1  # 10% gain = breakout

# -------------------------
# Feature Engineering
# -------------------------
def compute_features(df):
    df = df.copy()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=14).mean()
    df["Momentum"] = df["Close"].pct_change(periods=14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

# -------------------------
# Labeling: Breakout or Not
# -------------------------
def label_breakouts(df, future_days=5, threshold=0.1):
    df = df.copy()
    df["future_max"] = df["Close"].rolling(window=future_days).max().shift(-future_days)

    # Align before comparing
    future_max = df["future_max"].copy()
    close_shifted = df["Close"] * (1 + threshold)
    close_shifted.index = future_max.index

    df["label"] = (future_max > close_shifted).astype(int)
    df.dropna(inplace=True)
    return df

# -------------------------
# Training Pipeline
# -------------------------
print(f"Fetching {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
df = compute_features(df)
df = label_breakouts(df, threshold=BREAKOUT_THRESHOLD)

# Features and target
X = df[["Volume_Ratio", "Momentum", "RSI"]]
y = df["label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved!")
