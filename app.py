import streamlit as st
from scanner import scan_stocks

# Title and UI
st.set_page_config(page_title="Money Maker", layout="centered")

st.title("💸 Money Maker – AI Stock Breakout Assistant")

st.markdown("Enter stock tickers separated by commas (e.g. `AAPL, MSFT, NVDA`)")

tickers_input = st.text_input("Stock Tickers", value="AAPL, MSFT, NVDA, PLTR, AMZN, GOOGL, META, TSLA")

if st.button("🔍 Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if tickers:
        scan_stocks(tickers)
    else:
        st.error("Please enter at least one valid ticker symbol.")
