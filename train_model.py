import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from utils.preprocessing import preprocess_for_training

# === CONFIG ===
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG", "META", "NFLX", "ORCL",
    "BABA", "DIS", "BAC", "NKE", "CRM", "INTC", "CSCO", "IBM", "QCOM",
    "ADBE", "TXN", "AVGO", "PYPL", "AMZN", "WMT", "V", "MA", "JNJ", "PG",
    "XOM", "CVX", "KO", "PFE", "MRK", "T", "VZ", "MCD"
]
FUTURE_DAYS = 5
BREAKOUT_THRESHOLD = 1.10  # 10% rise = breakout

X_all = []
y_all = []

print("📊 Starting model training using yfinance...")

for ticker in TICKERS:
    print(f"📈 Fetching {ticker} from yfinance...")
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)

    # Fix for MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 50:
        print(f"⚠️ Skipping {ticker}: Not enough data.")
        continue

    # Create breakout target column
    df["future_max"] = df["Close"].rolling(window=FUTURE_DAYS).max().shift(-FUTURE_DAYS)
    df = df.dropna(subset=["future_max", "Close"])

    if df.empty:
        print(f"⚠️ Skipping {ticker}: Not enough labeled data.")
        continue

    try:
        X_scaled, df_processed = preprocess_for_training(df)

        df_processed = df_processed.tail(len(X_scaled))
        y = (df_processed["future_max"].values > df_processed["Close"].values * BREAKOUT_THRESHOLD).astype(int)

        if len(X_scaled) != len(y):
            print(f"❌ Length mismatch for {ticker}")
            continue

        X_all.extend(X_scaled)
        y_all.extend(y)

        print(f"✅ {ticker} processed: {len(y)} samples")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

if not X_all:
    raise RuntimeError("❌ No data available for training.")

# Check breakout label distribution
y_array = np.array(y_all)
unique, counts = np.unique(y_array, return_counts=True)
label_counts = dict(zip(unique, counts))
print("\n📊 Breakout label distribution in training data:")
print(label_counts)

# Sample some labels and features
print("\nSample training labels and features:")
for i in range(5):
    print(f"Label: {y_all[i]}, Features (first 5): {X_all[i][:5]}")

print("\n🧠 Training MLP model...")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, solver='adam', random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/breakout_model.pkl")
print("✅ Model saved to models/breakout_model.pkl")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")
