# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Constants
TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
BREAKOUT_THRESHOLD = 0.10  # 10% gain in next 7 days

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# --- 1. Fetch historical data ---
print(f"Fetching {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# --- 2. Feature engineering ---
df["returns"] = df["Close"].pct_change()
df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(14).mean()
df["momentum"] = df["Close"].pct_change(periods=14)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# --- 3. Label future breakouts ---
def label_breakouts(data, threshold=BREAKOUT_THRESHOLD):
    data = data.copy()
    data["future_max"] = data["Close"].shift(-1).rolling(7).max()
    data = data.dropna(subset=["future_max", "Close"])  # Ensure no missing values
    data["label"] = (data["future_max"] > data["Close"] * (1 + threshold)).astype(int)
    return data

df = label_breakouts(df)

# --- 4. Drop rows with missing features ---
df = df.dropna(subset=["volume_ratio", "momentum", "rsi"])

# --- 5. Prepare training data ---
X = df[["volume_ratio", "momentum", "rsi"]]
y = df["label"]

# --- 6. Scale and train model ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# --- 7. Save model and scaler ---
joblib.dump(model, "models/breakout_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved!")
