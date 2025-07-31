import pandas as pd
import joblib
import yfinance as yf
from utils.preprocessing import preprocess_single_stock
import os

# === CONFIG ===
MODEL_PATH = "models/breakout_model.pkl"
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "META", "NFLX", "ORCL",
    "BABA", "DIS", "BAC", "NKE", "CRM", "INTC", "CSCO", "IBM", "QCOM",
    "ADBE", "TXN", "AVGO", "PYPL", "AMZN", "WMT", "V", "MA", "JNJ", "PG",
    "XOM", "CVX", "KO", "PFE", "MRK", "T", "VZ", "MCD"
]
CONFIDENCE_THRESHOLD = 0.60

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Please train it with train_model.py first.")

model = joblib.load(MODEL_PATH)

results = []

print(f"📊 Scanning {len(DEFAULT_TICKERS)} stocks...\n")

for ticker in DEFAULT_TICKERS:
    print(f"🔎 Scanning {ticker}...")

    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if df.empty or len(df) < 50:
            print(f"⚠️ Skipping {ticker}: Not enough data.")
            continue

        X_scaled, df_processed = preprocess_single_stock(df)
        if len(X_scaled) == 0:
            print(f"⚠️ Skipping {ticker}: No valid features.")
            continue

        probs = model.predict_proba(X_scaled)
        if probs.ndim == 2:
            breakout_score = float(probs[-1, 1])
        else:
            breakout_score = float(probs[-1])

        if breakout_score >= CONFIDENCE_THRESHOLD:
            last_close_val = df_processed["Close"].values.flatten()[-1]
            last_close = float(last_close_val)

            result = {
                "Ticker": ticker,
                "Breakout Score": round(breakout_score, 4),
                "Last Close": round(last_close, 2)
            }
            results.append(result)

            print(f"✅ {ticker}: Score {breakout_score:.2f}")

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

if results:
    df_out = pd.DataFrame(results).sort_values(by="Breakout Score", ascending=False)
    df_out.to_csv("watchlist.csv", index=False)
    print("\n✅ Watchlist saved to watchlist.csv:")
    print(df_out)
else:
    print("\n❌ No breakouts detected today.")
