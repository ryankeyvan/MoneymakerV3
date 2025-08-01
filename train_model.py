#!/usr/bin/env python3
import os
import json
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# — adjust these paths if you like —
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")  # expect your CSVs here

os.makedirs(MODELS_DIR, exist_ok=True)

# how many trading days ahead to label as "breakout"
horizons = {
    "1w": 5,    # 1 week ≈ 5 trading days
    "1m": 22,   # 1 month ≈ 22 trading days
    "3m": 66    # 3 months ≈ 66 trading days
}

thresholds = {}

for h, days_ahead in horizons.items():
    # == load your pre-made dataset ==
    # we assume you have CSVs named like data/dataset_1w.csv
    # each with columns Open, High, Low, Close, Breakout (0/1 label for that horizon)
    src = os.path.join(DATA_DIR, f"dataset_{h}.csv")
    print(f"→ Loading {src}")
    df = pd.read_csv(src)

    # features & label
    X = df[["Open", "High", "Low", "Close"]].values
    y = df["Breakout"].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train a simple RF
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    print(f"→ Training model for {h}")
    model.fit(X_train, y_train)

    # save the model
    mpath = os.path.join(MODELS_DIR, f"breakout_model_{h}.pkl")
    with open(mpath, "wb") as f:
        pickle.dump(model, f)
    print(f"   saved model → {mpath}")

    # save test features & labels so evaluate.py can find them
    tf = pd.DataFrame(X_test, columns=["Open", "High", "Low", "Close"])
    lf = pd.DataFrame(y_test, columns=["Breakout"])
    feat_path = os.path.join(MODELS_DIR, f"test_features_{h}.csv")
    label_path = os.path.join(MODELS_DIR, f"test_labels_{h}.csv")
    tf.to_csv(feat_path, index=False)
    lf.to_csv(label_path, index=False)
    print(f"   saved test features → {feat_path}")
    print(f"   saved test labels   → {label_path}")

    # choose 0.5 as a default cutoff
    thresholds[h] = 0.5

# write out thresholds.json
tpath = os.path.join(MODELS_DIR, "thresholds.json")
with open(tpath, "w") as f:
    json.dump(thresholds, f, indent=2)
print(f"✔ Done. thresholds → {tpath}")
