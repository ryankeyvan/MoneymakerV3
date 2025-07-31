# ml_model.py

import numpy as np
import joblib
import os

# Load trained model and scaler from root directory
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ Trained model or scaler not found. Run train_model.py first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict_breakout(volume_ratio, momentum, rsi):
    # Prepare and scale the input features
    X = np.array([[volume_ratio, momentum, rsi]])
    X_scaled = scaler.transform(X)

    # Predict probability of breakout
    prob = model.predict_proba(X_scaled)[0][1]  # probability of breakout (class = 1)
    return round(prob, 3)  # keep 3 decimals for clarity
