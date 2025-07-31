import pandas as pd
import joblib
import yfinance as yf
from utils.preprocessing import preprocess_single_stock
import os

# === CONFIG ===
MODEL_PATH = "models/breakout_model.pkl"
TICKERS = ["NVDA"]  # Change this ticker as needed
CONFIDENCE_THRESHOLD = 0.4  # Lowered to catch more signals

# === Load ML model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Train it with train_model.py first.")

model = joblib.load(MODEL_PATH)

print("📊 Scanning stocks...\n")

for ticker in TICKERS:
    print(f"🔎 Scanning {ticker}...")

    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if df.empty or len(df) < 50:
            print(f"⚠️ Skipping {ticker}: Not enough data.")
            continue

        X_scaled, df_processed = preprocess_single_stock(df)

        if len(X_scaled) == 0:
            print(f"⚠️ Skipping {ticker}: No valid features.")
            continue

        probs = model.predict_proba(X_scaled)
        prob = float(probs[-1, 1])

        print(f"{ticker}: Breakout score {prob:.4f}")

        if prob >= CONFIDENCE_THRESHOLD:
            last_close = float(df_processed["Close"].values.flatten()[-1])
            print(f"✅ {ticker} PASSES threshold with score {prob:.4f}, Last Close {last_close:.2f}")

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")
