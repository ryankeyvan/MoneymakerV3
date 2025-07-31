import pandas as pd
import joblib
from alpha_vantage.timeseries import TimeSeries
from utils.preprocessing import preprocess_single_stock
from sklearn.exceptions import NotFittedError
import os

# === CONFIG ===
API_KEY = "JMOVPJWW0ZA4ASVW"
MODEL_PATH = "models/breakout_model.pkl"
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "META", "NFLX", "ORCL", "BABA", "DIS", "BAC", "NKE", "CRM"]
CONFIDENCE_THRESHOLD = 0.60

# === Alpha Vantage Setup ===
ts = TimeSeries(key=API_KEY, output_format='pandas')

def fetch_data(ticker):
    try:
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        data = data.sort_index()
        return data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"❌ Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

# === Load ML model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Train it with train_model.py first.")

model = joblib.load(MODEL_PATH)

# === Scan Loop ===
results = []

print("📊 Scanning stocks...\n")

for ticker in TICKERS:
    print(f"🔎 Scanning {ticker}...")

    df = fetch_data(ticker)

    if df.empty or len(df) < 50:
        print(f"⚠️ Skipping {ticker}: Not enough data.")
        continue

    try:
        X_scaled, df_processed = preprocess_single_stock(df)
        if len(X_scaled) == 0:
            print(f"⚠️ Skipping {ticker}: No valid features.")
            continue

        probs = model.predict_proba(X_scaled)
        breakout_score = probs[-1][1]  # Latest day

        if breakout_score >= CONFIDENCE_THRESHOLD:
            last_close = df_processed["Close"].iloc[-1]
            result = {
                "Ticker": ticker,
                "Breakout Score": round(breakout_score, 4),
                "Last Close": round(last_close, 2)
            }
            results.append(result)

            print(f"✅ {ticker}: Score {breakout_score:.2f}")

    except NotFittedError:
        print("❌ Model is not trained yet.")
    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

# === Output Results ===
if results:
    df_out = pd.DataFrame(results).sort_values(by="Breakout Score", ascending=False)
    df_out.to_csv("watchlist.csv", index=False)
    print("\n✅ Watchlist saved to watchlist.csv:")
    print(df_out)
else:
    print("\n❌ No breakouts detected today.")
