# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Simulated historical data (replace with real CSV if available)
data = pd.DataFrame({
    "volume_ratio": [1.2, 2.0, 0.8, 1.5, 2.5, 0.9, 1.0, 1.6, 2.1, 0.95],
    "price_momentum": [1.05, 1.2, 0.97, 1.15, 1.3, 0.95, 1.0, 1.18, 1.22, 0.98],
    "rsi": [55, 60, 45, 58, 70, 40, 50, 62, 65, 48],
    "breakout": [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]
})

X = data[["volume_ratio", "price_momentum", "rsi"]]
y = data["breakout"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "breakout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved!")
