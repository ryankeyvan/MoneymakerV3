import yfinance as yf
import joblib
import pandas as pd
from utils.preprocessing import preprocess_single_stock
from datetime import datetime

MODEL_PATH = "models/breakout_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def test_breakout_on_date(ticker, test_date_str):
    test_date = pd.to_datetime(test_date_str)

    print(f"📈 Fetching {ticker} data around {test_date_str}...")
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if test_date not in df.index:
        print(f"❌ Date {test_date_str} not in data for {ticker}.")
        return

    # Preprocess full dataframe
    X_scaled, df_processed = preprocess_single_stock(df)

    # Find index of test_date in df_processed
    if test_date not in df_processed.index:
        print(f"❌ Date {test_date_str} dropped during preprocessing.")
        return

    idx = df_processed.index.get_loc(test_date)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict breakout probability for that day
    prob = model.predict_proba(X_scaled)[idx][1]
    close_price = df_processed.loc[test_date, "Close"]

    print(f"{ticker} on {test_date_str}:")
    print(f"  Close Price: {close_price}")
    print(f"  Breakout Probability: {prob:.4f}")

if __name__ == "__main__":
    # Example: Replace with known breakout dates you want to test
    test_cases = [
        ("NVDA", "2023-05-15"),  # Hypothetical breakout date
        ("AAPL", "2023-01-25"),
        ("TSLA", "2023-03-10"),
    ]

    for ticker, date in test_cases:
        test_breakout_on_date(ticker, date)
        print("-" * 40)
