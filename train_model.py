# train_model.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import os

# 1) List of tickers to include in training:
TICKERS = [
    "AAPL","MSFT","NVDA","GOOG","AMZN",
    "TSLA","META","NFLX","IBM","ORCL",
    # …add more symbols here to improve your model
]

def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

def compute_features_and_labels(df, lookahead=5, breakout_pct=0.10):
    """
    Features: 
      - last day return
      - 5d avg return
      - 20d volatility
      - volume spike ratio
    Label = 1 if max return over next `lookahead` days ≥ breakout_pct, else 0
    """
    closes = df['Close']
    returns = closes.pct_change().fillna(0)
    vol = df['Volume']
    
    # features matrix
    feats = []
    labs = []
    for i in range(20, len(df) - lookahead):
        window = returns.iloc[i-20:i]
        feat = [
            returns.iloc[i],                     # last-day return
            window[-5:].mean(),                  # 5d avg
            window.std(),                        # 20d vol
            vol.iloc[i] / vol.iloc[i-5:i].mean() # volume spike
        ]
        future_max = (closes.iloc[i+1:i+1+lookahead] / closes.iloc[i] - 1).max()
        label = int(future_max >= breakout_pct)
        feats.append(feat)
        labs.append(label)
    return np.array(feats), np.array(labs)

# collect all tickers' data
all_X = []
all_y = []
for t in TICKERS:
    df = fetch_data(t)
    if df is None or df.empty: 
        print(f"⚠️  No data for {t}, skipping")
        continue
    X, y = compute_features_and_labels(df)
    all_X.append(X)
    all_y.append(y)

X = np.vstack(all_X)
y = np.concatenate(all_y)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train
model = MLPClassifier(
    hidden_layer_sizes=(50,25),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)

# evaluate & choose threshold by F1 on validation
probs = model.predict_proba(X_test)[:,1]
best_thresh = 0.0
best_f1 = 0
for thr in np.linspace(0,1,101):
    preds = (probs >= thr).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thr
print(f"🔍 Best validation F1 = {best_f1:.4f} at threshold = {best_thresh:.3f}")

# save model as binary pickle
os.makedirs("models", exist_ok=True)
with open("models/breakout_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved to models/breakout_model.pkl")
