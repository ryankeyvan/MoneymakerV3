import yfinance as yf

def debug_future_max(ticker, future_days=3):
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty:
        print(f"No data for {ticker}")
        return

    df["future_max"] = df["Close"].rolling(window=future_days).max().shift(-future_days)

    print(f"Columns: {df.columns.tolist()}")
    print(f"First 10 future_max values for {ticker}:\n{df['future_max'].head(10)}")
    print(f"Number of rows before dropna: {len(df)}")

    df = df.dropna(subset=["future_max", "Close"])

    print(f"Number of rows after dropna: {len(df)}")
    print(f"Tail after dropna:\n{df.tail(5)}")

if __name__ == "__main__":
    ticker = input("Enter ticker to test (e.g. ADBE): ").strip().upper()
    debug_future_max(ticker)
