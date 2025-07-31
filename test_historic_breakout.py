import yfinance as yf
import joblib
import pandas as pd
from utils.preprocessing import preprocess_single_stock

MODEL_PATH = "models/breakout_model.pkl"

def test_breakout_on_date(ticker, test_date_str):
    test_date = pd.to_datetime(test_date_str)

    print(f"📈 Fetching {ticker} data around {test_date_str}...")
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    # Find closest available date on or before test_date
    available_dates = df.index[df.index <= test_date]
    if len(available_dates) == 0:
        print(f"❌ No trading data available on or before {test_date_str} for {ticker}.")
        return
    closest_date = available_dates[-1]

    print(f"Using closest available date {closest_date.date()} instead of {test_date.date()}.")

    # Preprocess full dataframe
    X_scaled, df_processed = preprocess_single_stock(df)

    if closest_date not in df_processed.index:
        print(f"❌ Date {closest_date.date()} dropped during preprocessing.")
        return

    idx = df_processed.index.get_loc(closest_date)

    model = joblib.load(MODEL_PATH)
    prob = model.predict_proba(X_scaled)[idx][1]
    close_price = df_processed.loc[closest_date, "Close"]

    print(f"{ticker} on {closest_date.date()}:")
    print(f"  Close Price: {close_price}")
    print(f"  Breakout Probability: {prob:.4f}")

if __name__ == "__main__":
    test_cases = [
        ("NVDA", "2023-05-15"),
        ("AAPL", "2023-01-25"),
        ("TSLA", "2023-03-10"),
    ]

    for ticker, date in test_cases:
        test_breakout_on_date(ticker, date)
        print("-" * 40)
