import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ---------------------- CONFIG ----------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
LOOKAHEAD_DAYS = 5
BREAKOUT_THRESHOLD = 0.05
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ---------------------- FEATURE ENGINEERING ----------------------
def compute_features(df):
    df["Return"] = df["Close"].pct_change()
    df["Momentum"] = df["Return"].rolling(window=3).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=5).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# ---------------------- LABELING ----------------------
def label_breakouts(df, threshold=BREAKOUT_THRESHOLD, lookahead=LOOKAHEAD_DAYS):
    df = df.reset_index(drop=True)
    future_max = []
    for i in range(len(df)):
        future_window = df["Close"].iloc[i+1:i+1+lookahead]
        if not future_window.empty:
            future_max.append(future_window.max())
        else:
            future_max.append(np.nan)
    df["future_max"] = future_max
    df.dropna(subset=["future_max", "Close"], inplace=True)
    df["label"] = (df["future_max"] > df["Close"] * (1 + threshold)).astype(int)
    return df

# ---------------------- MAIN TRAINING ----------------------
all_data = []

for ticker in TICKERS:
    print(f"📈 Fetching {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        df = compute_features(df)
        df = label_breakouts(df)
        df = df.dropna(subset=["Momentum", "Volume_Ratio", "RSI"])
        features = df[["Volume_Ratio", "Momentum", "RSI"]]
        labels = df["label"]
        all_data.append((features, labels))
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

if not all_data:
    raise RuntimeError("No data was successfully processed.")

X = pd.concat([f for f, _ in all_data])
y = pd.concat([l for _, l in all_data])

# ---------------------- SCALING + TRAINING ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------- SAVE MODEL & SCALER ----------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("✅ Model and scaler saved!")
