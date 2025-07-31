# app.py

import streamlit as st
from scanner import scan_stocks
import pandas as pd
import time

st.set_page_config(page_title="📈 Money Maker AI", layout="wide", page_icon="💰")

st.title("💰 Money Maker AI Stock Breakout Scanner")
st.markdown("Scan live stock data and predict breakouts using AI.")

with st.sidebar:
    st.header("⚙️ Settings")
    default_watchlist = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "AMD", "NFLX", "INTC"]
    tickers = st.text_area("Enter Tickers (comma separated)", value=", ".join(default_watchlist)).split(",")
    tickers = [x.strip().upper() for x in tickers if x.strip()]
    run_scan = st.button("🔍 Run Scan")

if run_scan:
    with st.spinner("📡 Scanning..."):
        df, logs = scan_stocks(tickers)
        st.success("✅ Scan complete!")

        if not df.empty:
            st.subheader("📊 Scan Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name="scan_results.csv", mime="text/csv")

        st.subheader("📜 Scan Logs")
        for log in logs:
            st.write(log)
