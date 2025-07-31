# ml_model.py

import numpy as np
import joblib
import os

model_path = "models/breakout_model.pkl"
scaler_path = "models/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ Trained model or scaler not found. Run train_model.py first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict_breakout(volume_ratio, momentum, rsi):
    X = np.array([[volume_ratio, momentum, rsi]])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    return round(prob, 3)
