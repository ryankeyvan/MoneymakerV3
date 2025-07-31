# ml_model.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def predict_breakout(volume_ratio, price_momentum, rsi):
    # Dummy training data (historical patterns)
    X_train = np.array([
        [1.2, 1.05, 55],
        [2.0, 1.20, 60],
        [0.8, 0.97, 45],
        [1.5, 1.15, 58],
        [2.5, 1.30, 70],
        [0.9, 0.95, 40],
        [1.0, 1.00, 50]
    ])
    y_train = [1, 1, 0, 1, 1, 0, 0]  # 1 = breakout, 0 = no breakout

    # Step 1: Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 2: Train SVM model
    model = SVC(probability=True)
    model.fit(X_train_scaled, y_train)

    # Step 3: Prepare new input sample
    X_input = np.array([[volume_ratio, price_momentum, rsi]])  # shape (1, 3)

    # Step 4: Scale new sample using same scaler
    X_input_scaled = scaler.transform(X_input)

    # Step 5: Predict probability of breakout
    breakout_probability = model.predict_proba(X_input_scaled)[0][1]  # probability of class 1
    return breakout_probability
