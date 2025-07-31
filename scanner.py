import pandas as pd
import joblib
import yfinance as yf
from utils.preprocessing import preprocess_single_stock
from sklearn.exceptions import NotFittedError
import os

# === CONFIG ===
MODEL_PATH = "models/breakout_model.pkl"
TICKERS = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "META", "NFLX",
    "ORCL", "BABA", "DIS", "BAC", "NKE", "CRM", "INTC", "CSCO",
    "IBM", "QCOM", "ADBE", "TXN", "AVGO", "PYPL", "AMZN", "WMT",
    "V", "MA", "JNJ", "PG", "XOM", "CVX", "KO", "PFE", "MRK",
    "T", "VZ", "MCD"
]
CONFIDENCE_THRESHOLD = 0.60

# === Load ML model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Train it with train_model.py first.")

model = joblib.load(MODEL_PATH)

results = []

print("📊 Scanning stocks...\n")

for ticker in TICKERS:
    print(f"🔎 Scanning {ticker}...")

    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)

        # Flatten MultiIndex columns if they exist
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
        prob = float(probs[-1, 1])  # Extract scalar breakout probability

        if prob >= CONFIDENCE_THRESHOLD:
            last_close = float(df_processed["Close"].values.flatten()[-1])

            print(f"✅ {ticker}: Score {prob:.4f}, Last Close {last_close:.2f}")

            results.append({
                "Ticker": ticker,
                "Breakout Score": round(prob, 4),
                "Last Close": round(last_close, 2)
            })

    except NotFittedError:
        print("❌ Model is not trained yet.")
        break
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

if results:
    df_out = pd.DataFrame(results).sort_values(by="Breakout Score", ascending=False)
    df_out.to_csv("watchlist.csv", index=False)
    print("\n✅ Watchlist saved to watchlist.csv:")
    print(df_out)
else:
    print("\n❌ No breakouts detected today.")
