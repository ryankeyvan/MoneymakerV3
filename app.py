import streamlit as st
import pandas as pd
from scanner import run_breakout_scan
from utils.charting import plot_stock_chart

st.set_page_config(page_title="Money Maker", layout="wide")
st.title("💸 Money Maker – AI Stock Breakout Assistant")

# Load watchlist
watchlist = pd.read_csv("watchlist.csv")
tickers = watchlist["Ticker"].tolist()

st.sidebar.subheader("🔍 Settings")
run_scan = st.sidebar.button("Run Breakout Scan")

if run_scan:
    st.write("🔄 Scanning...")
    st.write(f"Scanning {len(tickers)} tickers...")
    results = run_breakout_scan(tickers)
    st.write(f"Scan complete. Found {len(results)} results.")

    for result in results:
        ticker = result["Ticker"]
        st.subheader(f"📈 {ticker} — {result['Signal']}")
        st.markdown(f"**Breakout Score:** {result['Breakout Score']:.2f}")
        st.markdown(f"**Target Price (1M):** ${result['Target Price']:.2f}")
        st.markdown(f"**Stop Loss:** ${result['Stop Loss']:.2f}")
        st.markdown(f"**Sentiment Score:** {result['Sentiment']}")

        st.pyplot(plot_stock_chart(ticker))
        st.markdown("---")
