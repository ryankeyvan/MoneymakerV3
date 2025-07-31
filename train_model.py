import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from utils.preprocessing import preprocess_for_training

# === CONFIG ===
API_KEY = "JMOVPJWW0ZA4ASVW"
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG", "META", "NFLX", "ORCL",
    "BABA", "DIS", "BAC", "NKE", "CRM", "INTC", "CSCO", "IBM", "QCOM",
    "ADBE", "TXN", "AVGO", "PYPL", "AMZN", "WMT", "V", "MA", "JNJ", "PG",
    "XOM", "CVX", "KO", "PFE", "MRK", "T", "VZ", "MCD"
]
FUTURE_DAYS = 5
BREAKOUT_THRESHOLD = 1.10  # 10% rise = breakout

ts = TimeSeries(key=API_KEY, output_format='pandas')

def fetch_data_alpha(ticker):
    print(f"📈 Fetching {ticker} from Alpha Vantage...")
    try:
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
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
        print(f"❌ Error fetching {ticker}: {e}")
        return pd.DataFrame()

X_all = []
y_all = []

print("📊 Starting model training...")

for ticker in TICKERS:
    df = fetch_data_alpha(ticker)

    if df.empty:
        print(f"⚠️ Skipping {ticker}: No usable data.")
        continue

    df["future_max"] = df["Close"].rolling(window=FUTURE_DAYS).max().shift(-FUTURE_DAYS)
    df = df.dropna(subset=["future_max", "Close"])

    if df.empty:
        print(f"⚠️ Skipping {ticker}: Not enough labeled data.")
        continue

    try:
        X_scaled, df_processed = preprocess_for_training(df)

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
