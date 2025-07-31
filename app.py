import streamlit as st
import pandas as pd
from scanner import scan_stocks

# Try optional import
try:
    from scanner import get_top_100_stocks
    HAS_AUTO_SCAN = True
except ImportError:
    HAS_AUTO_SCAN = False

st.set_page_config(page_title="📈 Money Maker AI", layout="wide")
st.title("💸 Money Maker AI - Breakout Stock Scanner")

# Sidebar setup
st.sidebar.header("📊 Scanner Options")
mode_options = ["Enter Tickers"]
if HAS_AUTO_SCAN:
    mode_options.append("Auto Scan 100+")
mode = st.sidebar.radio("Select Scan Mode", mode_options)

if mode == "Enter Tickers" or not HAS_AUTO_SCAN:
    tickers_input = st.sidebar.text_area("Enter Tickers (comma separated)", "AAPL, TSLA, NVDA, MSFT")
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
else:
    tickers = get_top_100_stocks()

scan_button = st.sidebar.button("🚀 Run Scan")

# Key/Legend
with st.expander("📘 Indicator Key"):
    st.markdown("""
    - **Breakout Score**: Confidence (0–100) of breakout potential.
    - **RSI**: Relative Strength Index (30–70 = neutral; >70 = overbought).
    - **Momentum**: 10-day percent price change.
    - **Volume Change**: Spike vs. average volume (%).
    - **Projected 1M Price**: Estimated target based on recent momentum and volume.
    """)

progress = st.empty()
log_area = st.empty()
output_area = st.empty()

if scan_button:
    st.session_state["results"] = []

    progress_bar = progress.progress(0, text="🔎 Starting scan...")

    df = scan_stocks(
        tickers=tickers,
        auto=(mode == "Auto Scan 100+"),
        update_progress=lambda p: progress_bar.progress(p, text=f"🔍 Scanning... {int(p * 100)}%"),
    )

    if df is None or df.empty:
        output_area.warning("⚠️ No valid breakout scores found or no data available.")
    else:
        df = df.sort_values(by="Breakout Score", ascending=False)
        st.session_state["results"] = df
        output_area.dataframe(df)

if "results" not in st.session_state or st.session_state["results"] is None:
    st.info("⚠️ No results found or scan hasn’t been run yet.")
