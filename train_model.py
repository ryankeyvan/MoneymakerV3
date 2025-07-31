import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Constants
TICKER = "AAPL"  # You can swap this with more tickers in the future
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
BREAKOUT_THRESHOLD = 0.1  # 10% breakout
FUTURE_DAYS = 5

def label_breakouts(df, future_days=FUTURE_DAYS, threshold=BREAKOUT_THRESHOLD):
    df = df.copy()

    if len(df) < future_days + 1:
        raise ValueError("❌ Not enough data to compute future breakout labels.")

    # Calculate future max
    df["future_max"] = df["Close"].shift(-1).rolling(window=future_days).max()

    # Drop missing values
    df = df.dropna(subset=["future_max", "Close"])

    # Compute labels (1 if price jumps 10% above current close)
    breakout_label = (df["future_max"].values > (df["Close"].values * (1 + threshold))).astype(int)
    df["label"] = breakout_label

    return df

def compute_features(df):
    df["momentum"] = df["Close"].pct_change(periods=14)
    df["volume_avg"] = df["Volume"].rolling(window=14).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_avg"]

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df.dropna()

# Step 1: Download Data
print(f"Fetching {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Step 2: Label Breakouts
df = label_breakouts(df, threshold=BREAKOUT_THRESHOLD)

# Step 3: Compute Features
df = compute_features(df)

# Step 4: Prepare Model Data
features = df[["volume_ratio", "momentum", "rsi"]]
labels = df["label"]

# Step 5: Scale and Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Save Model and Scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/breakout_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved!")
