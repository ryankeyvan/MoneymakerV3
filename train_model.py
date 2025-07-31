# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Setup
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
BREAKOUT_THRESHOLD = 0.05
LOOKAHEAD_DAYS = 5
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def label_breakouts(df, threshold=BREAKOUT_THRESHOLD, lookahead=LOOKAHEAD_DAYS):
    df = df.copy()
    df["future_max"] = df["Close"].shift(-lookahead).rolling(lookahead).max()
    df.dropna(subset=["future_max", "Close"], inplace=True)
    df["label"] = (df["future_max"] > df["Close"] * (1 + threshold)).astype(int)
    return df

def compute_features(df):
    df = df.copy()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["Price_Momentum"] = df["Close"] / df["Close"].shift(5)
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() /
                              df["Close"].pct_change().rolling(14).std()))
    df = df[["Volume_Ratio", "Price_Momentum", "RSI", "label"]].dropna()
    return df

# Aggregate data
all_data = []
for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    if df.empty:
        print(f"⚠️ Skipping {ticker} due to empty data.")
        continue
    try:
        df = label_breakouts(df)
        features = compute_features(df)
        all_data.append(features)
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

# Combine and train
if not all_data:
    raise RuntimeError("No data was successfully processed.")
df_all = pd.concat(all_data)
X = df_all.drop(columns=["label"])
y = df_all["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, os.path.join(MODEL_DIR, "breakout_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("\n✅ Model and scaler saved in /models")
