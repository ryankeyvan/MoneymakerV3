import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle, json, os

# Tickers for model training; extend as needed
TICKERS = [
    "AAPL","MSFT","NVDA","GOOG","AMZN",
    "TSLA","META","NFLX","IBM","ORCL",
    # ... add more symbols to strengthen training
]

# Horizon definitions: lookahead days and breakout threshold pct
HORIZONS = {
    "1w": {"lookahead": 5, "pct": 0.10},
    "1m": {"lookahead": 21, "pct": 0.15},
    "3m": {"lookahead": 63, "pct": 0.30},
}

# Ensure models directory
os.makedirs("models", exist_ok=True)

# Fetch price history
def fetch_data(ticker, period="1y", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

# Compute features & labels for each horizon
def compute_features_and_labels(df, lookahead, breakout_pct):
    closes = df['Close']
    returns = closes.pct_change().fillna(0)
    vol = df['Volume']
    feats, labs = [], []
    for i in range(20, len(df) - lookahead):
        window = returns.iloc[i-20:i]
        feat = [
            returns.iloc[i],                   # last-day return
            window[-5:].mean(),                # 5-day average return
            window.std(),                      # 20-day volatility
            vol.iloc[i] / vol.iloc[i-5:i].mean()  # volume spike ratio
        ]
        future_max = (closes.iloc[i+1:i+1+lookahead] / closes.iloc[i] - 1).max()
        label = int(future_max >= breakout_pct)
        feats.append(feat)
        labs.append(label)
    return np.array(feats), np.array(labs)

# Collect and train models per horizon
def train_models():
    thresholds = {}
    for name, cfg in HORIZONS.items():
        all_X, all_y = [], []
        for ticker in TICKERS:
            df = fetch_data(ticker)
            if df is None or df.empty:
                print(f"⚠️ Skipping {ticker} for {name}: no data")
                continue
            X, y = compute_features_and_labels(df, cfg['lookahead'], cfg['pct'])
            if X.size and y.size:
                all_X.append(X)
                all_y.append(y)
        if not all_X:
            print(f"⚠️ No training data for horizon {name}")
            continue
        X = np.vstack(all_X).reshape(-1, all_X[0].shape[1])
        y = np.concatenate(all_y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = MLPClassifier(hidden_layer_sizes=(50,25), activation='relu', solver='adam',
                              max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        best_thr, best_f1 = 0.0, 0.0
        for thr in np.linspace(0,1,101):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(y_test, preds)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thresholds[name] = best_thr
        with open(f"models/breakout_{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Trained {name}: F1={best_f1:.4f} @ thr={best_thr:.3f}")
    # Save thresholds
    with open("models/thresholds.json", "w") as f:
        json.dump(thresholds, f)
    print("✅ All models trained and thresholds saved.")

if __name__ == "__main__":
    train_models()
