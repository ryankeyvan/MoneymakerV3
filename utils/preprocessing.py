import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import joblib
import os

SCALER_PATH = os.path.join("models", "scaler.pkl")

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)

    # Technical indicators
    df["rsi"] = RSIIndicator(df["Close"], window=14).rsi()
    df["macd"] = MACD(df["Close"]).macd_diff()
    df["ema_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(df["Close"])
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()

    # Volume ratios
    df["vol_change_1d"] = df["Volume"].pct_change(1)
    df["vol_ma_5"] = df["Volume"].rolling(window=5).mean()
    df["vol_to_avg"] = df["Volume"] / (df["vol_ma_5"] + 1e-5)

    # Price position relative to moving averages
    df["price_above_ema20"] = df["Close"] > df["ema_20"]
    df["price_above_ema50"] = df["Close"] > df["ema_50"]
    df["price_above_bbm"] = df["Close"] > df["bb_bbm"]

    # Drop incomplete rows
    df = df.dropna()

    return df

def get_feature_columns():
    return [
        "return_1d", "return_5d", "rsi", "macd",
        "bb_bbm", "bb_bbh", "bb_bbl",
        "ema_20", "ema_50",
        "vol_change_1d", "vol_to_avg",
        "price_above_ema20", "price_above_ema50", "price_above_bbm"
    ]

def ensure_2d_array(X):
    X = np.array(X)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def safe_scalar_from_series(series):
    # Extracts a scalar float from a pandas Series/array safely
    arr = series.values.flatten()
    return float(arr[-1])

def preprocess_for_training(df: pd.DataFrame):
    df = generate_features(df)
    features = df[get_feature_columns()]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = ensure_2d_array(X_scaled)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, df

def preprocess_single_stock(df: pd.DataFrame):
    df = generate_features(df)
    features = df[get_feature_columns()]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(features)
    X_scaled = ensure_2d_array(X_scaled)

    return X_scaled, df.tail(len(X_scaled))
