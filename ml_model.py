# ml_model.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def predict_breakout(volume_ratio, price_momentum, rsi):
    # Dummy training data
    X_train = np.array([
        [1.2, 1.05, 55],
        [2.0, 1.20, 60],
        [0.8, 0.97, 45],
        [1.5, 1.15, 58],
        [2.5, 1.3, 70],
        [0.9, 0.95, 40],
        [1.0, 1.0, 50]
    ])
    y_train = [1, 1, 0, 1, 1, 0, 0]

    # Scale training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM model
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)

    # Ensure input shape is 2D (1 sample, 3 features)
    X_input = np.array([volume_ratio, price_momentum, rsi]).reshape(1, -1)

    # Scale input
    X_input_scaled = scaler.transform(X_input)

    # Predict probability of breakout
    breakout_prob = model.predict_proba(X_input_scaled)[0][1]
    return breakout_prob
