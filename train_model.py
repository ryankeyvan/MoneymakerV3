import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# === 1. Create synthetic historical-like breakout dataset ===
# Features: [volume_ratio, price_momentum, rsi]
# Target: 1 = breakout, 0 = no breakout
data = {
    "volume_ratio": np.random.normal(loc=1.5, scale=0.5, size=1000),
    "price_momentum": np.random.normal(loc=1.1, scale=0.15, size=1000),
    "rsi": np.random.normal(loc=55, scale=10, size=1000),
}

df = pd.DataFrame(data)

# Simulate labels: higher volume + high momentum + RSI ~ breakout
df["target"] = (
    (df["volume_ratio"] > 1.4)
    & (df["price_momentum"] > 1.05)
    & (df["rsi"] > 50)
).astype(int)

# === 2. Train-test split ===
X = df[["volume_ratio", "price_momentum", "rsi"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === 4. Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 5. Save model and scaler ===
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved!")
