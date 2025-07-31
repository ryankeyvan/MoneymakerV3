import numpy as np

def predict_breakout(features_scaled):
    """
    Simulate breakout prediction using a weighted formula.
    Replace with your trained ML model if available.
    """

    # Assuming features_scaled shape: (1, 5) → [rsi, macd, obv, momentum, volume_change]
    weights = np.array([0.15, 0.2, 0.1, 0.25, 0.3])  # Sum to 1
    score = np.dot(features_scaled[0], weights)

    # Ensure it's between 0 and 1
    return max(0.0, min(1.0, score))
