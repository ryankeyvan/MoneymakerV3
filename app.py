# app.py

import streamlit as st
from scanner import scan_stocks
import pandas as pd

st.set_page_config(page_title="📈 Money Maker AI", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 Money Maker AI - Stock Breakout Scanner</h1>", unsafe_allow_html=True)

default_watchlist = ["AAPL", "TSLA", "AMD", "NVDA", "MSFT", "GOOGL", "CRM", "PYPL", "UBER", "DIS", "SHOP", "NFLX"]
tickers = st.text_input("Enter comma-separated tickers:", ",".join(default_watchlist))
ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

if st.button("🔍 Scan for Breakouts"):
    with st.spinner("Scanning..."):
        results_df, logs = scan_stocks(ticker_list)

    if not results_df.empty:
        st.success("✅ Scan complete!")
        st.dataframe(results_df.sort_values(by="Breakout Score", ascending=False), use_container_width=True)
    else:
        st.warning("No valid breakout data found.")

    if logs:
        with st.expander("📜 Scan Logs"):
            for log in logs:
                st.write(log)
