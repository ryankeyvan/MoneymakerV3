# ml_model.py
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def predict_breakout(volume_ratio, price_momentum, rsi):
    # Sample training data for demo (features = [volume_ratio, price_momentum, rsi])
    X_train = np.array([
        [1.2, 1.05, 55], [2.0, 1.20, 60], [0.8, 0.97, 45],
        [1.5, 1.15, 58], [2.5, 1.3, 70], [0.9, 0.95, 40],
        [1.0, 1.0, 50]
    ])
    y_train = [1, 1, 0, 1, 1, 0, 0]  # 1 = breakout, 0 = no breakout

    # New data point
    X_test = np.array([[volume_ratio, price_momentum, rsi]])

    # Scale features for consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM classifier for simple decision boundary
    model = SVC(probability=True)
    model.fit(X_scaled, y_train)

    # Predict breakout probability (confidence score between 0 and 1)
    prob = model.predict_proba(X_test_scaled)[0][1]
    return prob
