import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scanner import scan_tickers

# App configuration
st.set_page_config(page_title="MoneyMakerV3 AI Stock Breakout Assistant", layout="wide")
st.title("💰 MoneyMakerV3 AI Stock Breakout Assistant")

# Session state for watchlist persistence
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# Sidebar: Watchlist management
st.sidebar.header("🍽️ Watchlist")
ticker_input = st.sidebar.text_input("Enter tickers (comma separated)")
if st.sidebar.button("Add to Watchlist"):
    new = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist + new))
    st.sidebar.success(f"Added: {', '.join(new)}")
if st.sidebar.button("Clear Watchlist"):
    st.session_state.watchlist = []
    st.sidebar.info("Watchlist cleared.")

st.sidebar.markdown("**Current Watchlist:**")
st.sidebar.write(st.session_state.watchlist)

# Main scanning interface
st.subheader("📈 Stock Scanner")
use_watchlist = st.checkbox("Use watchlist", value=True)
if use_watchlist:
    tickers = st.session_state.watchlist
else:
    tickers = [t.strip().upper() for t in st.text_input("Tickers to scan (comma separated)").split(',') if t.strip()]

if st.button("Run Scan"):
    if not tickers:
        st.warning("Please add at least one ticker to scan.")
    else:
        # Run scanner
        with st.spinner("Scanning stocks, please wait..."):
            results, failures = scan_tickers(tickers)

        df = pd.DataFrame(results)
        st.subheader("🔍 Scan Results")
        st.dataframe(df)

        # Downloadable CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download results as CSV", csv_data,
            file_name=f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv'
        )

        # Display failures
        if failures:
            st.subheader("⚠️ Failures")
            st.write(failures)

        # Price chart section
        st.subheader("📊 Price Chart")
        selected = st.selectbox(
            "Select a ticker to view its 6-month price history", 
            df['ticker'].tolist() if not df.empty else []
        )
        if selected:
            hist = yf.download(selected, period="6mo", interval="1d", progress=False)
            if not hist.empty:
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist['Close'], linewidth=1.5)
                ax.set_title(f"{selected} Price (6mo)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price ($)")
                st.pyplot(fig)
            else:
                st.info("No historical data available for selected ticker.")
