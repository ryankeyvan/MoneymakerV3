import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from utils.preprocessing import preprocess_single_stock
import os

# === CONFIG ===
MODEL_PATH = "models/breakout_model.pkl"
TICKERS = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOG", "META",
    "NFLX", "ORCL", "BABA", "DIS", "BAC", "NKE", "CRM"
]
CONFIDENCE_THRESHOLD = 0.60

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Please train it using train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# === Streamlit UI ===
st.set_page_config(page_title="MoneyMakerV3", layout="wide")
st.title("📈 MoneyMakerV3 — Breakout Stock Scanner")
st.markdown("Scans tickers and ranks breakout candidates using your trained AI model.")

if st.button("🚀 Scan Now"):
    results = []

    with st.spinner("Scanning for breakouts using yfinance live data..."):
        for ticker in TICKERS:
            try:
                df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

                if df.empty or len(df) < 50:
                    continue

                X_scaled, df_processed = preprocess_single_stock(df)
                if len(X_scaled) == 0:
                    continue

                probs = model.predict_proba(X_scaled)
                if probs.ndim == 2:
                    prob = float(probs[-1, 1])
                else:
                    prob = float(probs[-1])

                if prob >= CONFIDENCE_THRESHOLD:
                    last_close_val = df_processed["Close"].values.flatten()[-1]
                    last_close = float(last_close_val)

                    results.append({
                        "Ticker": ticker,
                        "Breakout Score": round(prob, 4),
                        "Last Close": round(last_close, 2)
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
