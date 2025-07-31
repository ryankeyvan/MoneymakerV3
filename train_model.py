import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve

from utils.preprocessing import preprocess_for_training

# === CONFIG ===
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG", "META", "NFLX", "ORCL",
    "BABA", "DIS", "BAC", "NKE", "CRM", "INTC", "CSCO", "IBM", "QCOM",
    "ADBE", "TXN", "AVGO", "PYPL", "AMZN", "WMT", "V", "MA", "JNJ", "PG",
    "XOM", "CVX", "KO", "PFE", "MRK", "T", "VZ", "MCD"
]
FUTURE_DAYS = 5
BREAKOUT_THRESHOLD = 1.10  # used to create labels (10% rise)
MODEL_PATH = "models/breakout_model.pkl"

# === FETCH & LABEL ===
X_all, y_all = [], []

print("📊 Gathering and labeling data...")
for ticker in TICKERS:
    print(f"  • {ticker}", end="…", flush=True)
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < FUTURE_DAYS + 2:
        print(" skipped (not enough data)")
        continue

    # create breakout label
    df["future_max"] = df["Close"].rolling(window=FUTURE_DAYS).max().shift(-FUTURE_DAYS)
    df = df.dropna(subset=["future_max", "Close"])
    if df.empty:
        print(" skipped (no labeled data)")
        continue

    # extract features and labels
    X_scaled, df_proc = preprocess_for_training(df)
    df_proc = df_proc.tail(len(X_scaled))
    labels = (df_proc["future_max"].values
              > df_proc["Close"].values * BREAKOUT_THRESHOLD).astype(int)

    if len(labels) == len(X_scaled):
        X_all.extend(X_scaled)
        y_all.extend(labels)
        print(f" {len(labels)} samples")
    else:
        print(" skipped (length mismatch)")

if not X_all:
    raise RuntimeError("❌ No data available for training.")

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# === MODEL TRAINING ===
print("\n🧠 Training MLPClassifier…")
model = MLPClassifier(hidden_layer_sizes=(128, 64),
                      max_iter=500, solver="adam", random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# === EVALUATION: ACCURACY ===
acc = accuracy_score(y_test, model.predict(X_test))
print(f"🎯 Test accuracy: {acc*100:.2f}%")

# === THRESHOLD SWEEP FOR PRECISION/RECALL/F1 ===
probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)

best_idx = np.argmax(f1s)
best_thresh = thresholds[best_idx]
best_f1 = f1s[best_idx]
best_prec = precisions[best_idx]
best_rec = recalls[best_idx]

print("\n🔧 Threshold tuning results:")
print(f"  ▶ Best by F1 = {best_f1:.3f} at threshold = {best_thresh:.3f}")
print(f"    Precision = {best_prec:.3f}, Recall = {best_rec:.3f}")

print("\nTop 5 thresholds by F1:")
top5 = np.argsort(f1s)[-5:][::-1]
for i in top5:
    print(f"    thresh={thresholds[i]:.3f} → F1={f1s[i]:.3f}")

