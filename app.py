import streamlit as st
import pandas as pd
import joblib
from alpha_vantage.timeseries import TimeSeries
from utils.preprocessing import preprocess_single_stock
import matplotlib.pyplot as plt
import os

# === CONFIG ===
API_KEY = "JMOVPJWW0ZA4ASVW"
MODEL_PATH = "models/breakout_model.pkl"
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "META", "NFLX", "ORCL", "BABA", "DIS", "BAC", "NKE", "CRM"]
CONFIDENCE_THRESHOLD = 0.60

ts = TimeSeries(key=API_KEY, output_format='pandas')

# === Load Model ===
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="MoneyMakerV3", layout="wide")
st.title("📈 MoneyMakerV3 — Breakout Stock Scanner")
st.markdown("Scans top tickers and ranks breakout candidates using AI.")

if st.button("🚀 Scan Now"):
    results = []

    with st.spinner("Scanning for breakouts..."):
        for ticker in TICKERS:
            try:
                data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                }, inplace=True)
                data = data.sort_index()
                df = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

                if df.empty or len(df) < 50:
                    continue

                X_scaled, df_processed = preprocess_single_stock(df)
                if len(X_scaled) == 0:
                    continue

                prob = model.predict_proba(X_scaled)[-1][1]  # Most recent
                if prob >= CONFIDENCE_THRESHOLD:
                    results.append({
                        "Ticker": ticker,
                        "Breakout Score": round(prob, 4),
                        "Last Close": round(df_processed["Close"].iloc[-1], 2)
                    })

            except Exception as e:
                st.error(f"{ticker} error: {e}")

    if results:
        df_out = pd.DataFrame(results).sort_values(by="Breakout Score", ascending=False)
        st.success(f"✅ {len(df_out)} breakout candidates found.")
        st.dataframe(df_out)

        csv = df_out.to_csv(index=False).encode()
        st.download_button("📥 Download Watchlist (CSV)", csv, file_name="watchlist.csv")
    else:
        st.warning("❌ No breakouts found today.")
