import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score
import joblib
import os
from utils.preprocessing import preprocess_for_training

# === CONFIG ===
TICKERS = [ ... same list as before ... ]
FUTURE_DAYS = 5
BREAKOUT_THRESHOLD = 1.10  # unchanged
MODEL_PATH = "models/breakout_model.pkl"

# === FETCH & LABEL ===
X_all, y_all = [], []
for ticker in TICKERS:
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=False)
    # flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 50:
        continue

    df["future_max"] = df["Close"].rolling(FUTURE_DAYS).max().shift(-FUTURE_DAYS)
    df = df.dropna(subset=["future_max", "Close"])
    if df.empty:
        continue

    X_scaled, df_proc = preprocess_for_training(df)
    df_proc = df_proc.tail(len(X_scaled))
    labels = (df_proc["future_max"].values >
              df_proc["Close"].values * BREAKOUT_THRESHOLD).astype(int)

    if len(labels) == len(X_scaled):
        X_all.extend(X_scaled)
        y_all.extend(labels)

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)

# === MODEL TRAIN ===
model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)
model.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

# === EVALUATE ACCURACY ===
acc = accuracy_score(y_test, model.predict(X_test))
print(f"🎯 Overall accuracy: {acc*100:.2f}%")

# === THRESHOLD TUNING ===
probs = model.predict_proba(X_test)[:,1]
prec, rec, thresh = precision_recall_curve(y_test, probs)
f1_scores = 2 * prec * rec / (prec + rec + 1e-12)

best_idx = np.argmax(f1_scores)
best_thresh = thresh[best_idx]
best_f1    = f1_scores[best_idx]
best_prec  = prec[best_idx]
best_rec   = rec[best_idx]

print("\n🔧 Threshold tuning results:")
print(f" Best threshold by F1: {best_thresh:.3f}")
print(f"   ▶ F1 = {best_f1:.3f}, Precision = {best_prec:.3f}, Recall = {best_rec:.3f}")

# (Optional) print a few top candidates for thresholds
print("\nThresholds vs. F1 (top 5):")
top5 = np.argsort(f1_scores)[-5:][::-1]
for i in top5:
    print(f"  thresh={thresh[i]:.3f} → F1={f1_scores[i]:.3f}")
