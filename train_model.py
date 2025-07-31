import yfinance as yf
import pandas as pd
import numpy as np
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
FUTURE_DAYS = 3
BREAKOUT_THRESHOLD = 1.10  # 10% rise = breakout

def fetch_and_prepare(ticker, future_days=FUTURE_DAYS):
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False)

    # Flatten multi-index columns if they exist (fixes KeyError on columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty or len(df) < future_days + 1:
        print(f"⚠️ Not enough data for {ticker}")
        return None

    df["future_max"] = df["Close"].rolling(window=future_days).max().shift(-future_days)
    df = df.dropna(subset=["future_max", "Close"])

    if df.empty:
        print(f"⚠️ No valid labeled data for {ticker} after processing")
        return None

    return df

X_all = []
y_all = []

print("📊 Starting model training using yfinance...")

for ticker in TICKERS:
    print(f"📈 Fetching {ticker} from yfinance...")
    try:
        df = fetch_and_prepare(ticker)
        if df is None:
            continue

        # Preprocess features and get processed df
        X_scaled, df_processed = preprocess_for_training(df)

        # Align y with X_scaled length
        df_processed = df_processed.tail(len(X_scaled))

        y = (df_processed["future_max"].values.flatten() > df_processed["Close"].values.flatten() * BREAKOUT_THRESHOLD).astype(int)

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

print("\n🧠 Training MLP model...")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/breakout_model.pkl")
print("✅ Model saved to models/breakout_model.pkl")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")
