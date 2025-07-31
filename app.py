import pandas as pd
import streamlit as st
from scanner import run_breakout_scan
from utils.charting import plot_stock_chart

# Set Streamlit config
st.set_page_config(page_title="Money Maker", layout="wide")
st.title("💸 Money Maker – AI Stock Breakout Assistant")

# Load watchlist
try:
    watchlist = pd.read_csv("watchlist.csv")
    tickers = watchlist["Ticker"].tolist()
except Exception as e:
    st.error(f"⚠️ Failed to load watchlist.csv: {e}")
    st.stop()

# Sidebar UI
st.sidebar.subheader("🔍 Settings")
run_scan = st.sidebar.button("Run Breakout Scan")

# Run scan logic
if run_scan:
    st.write("🌀 Scanning...")
    st.write(f"Scanning {len(tickers)} tickers...")
    results = run_breakout_scan(tickers)
    st.write(f"✅ Scan complete. Found {len(results)} results.")

    # Display results
    for result in results:
        ticker = result.get("Ticker", "N/A")
        st.subheader(f"📉 {ticker} — {result.get('Signal', 'N/A')}")

        try:
            st.markdown(f"**Breakout Score:** {result['Breakout Score']:.2f}")
        except:
            st.markdown("**Breakout Score:** N/A")

        try:
            st.markdown(f"**Target Price (1M):** ${result['Target Price']:.2f}")
        except:
            st.markdown("**Target Price (1M):** N/A")

        try:
            st.markdown(f"**Stop Loss:** ${result['Stop Loss']:.2f}")
        except:
            st.markdown("**Stop Loss:** N/A")

        st.markdown(f"**Sentiment Score:** {result.get('Sentiment', 'N/A')}")

        # Optional: Display chart
        try:
            plot_stock_chart(ticker)
        except:
            st.warning("⚠️ Failed to render stock chart.")
