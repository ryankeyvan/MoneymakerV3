import streamlit as st
import pandas as pd
from scanner import scan_stocks, get_all_stocks_above_5_dollars

st.set_page_config(page_title="📈 Money Maker AI", layout="wide")
st.title("💸 Money Maker AI - Breakout Stock Scanner")

# Sidebar Options
st.sidebar.header("📊 Scanner Options")
mode = st.sidebar.radio("Choose Scan Mode:", ["Enter Tickers", "Auto Scan 100+"])
if mode == "Enter Tickers":
    ticker_input = st.sidebar.text_area("Enter Tickers (comma separated)", "AAPL, TSLA, NVDA")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
else:
    tickers = get_all_stocks_above_5_dollars()

scan_button = st.sidebar.button("🚀 Run Scan")

# Indicator Legend
with st.expander("📘 Indicator Key"):
    st.markdown("""
    - **Breakout Score**: Confidence score (0–1) of breakout potential. >0.7 = strong.
    - **RSI**: Relative Strength Index (30–70 = neutral).
    - **Momentum**: 14-day % price movement.
    - **Volume Change**: Spike compared to average volume (%).
    - **Signal**: 🔥 Buy / 🧐 Watch
    """)

progress = st.empty()
log_area = st.empty()
output_area = st.empty()

if scan_button:
    st.session_state["results"] = []
    st.session_state["logs"] = []

    progress_bar = progress.progress(0, text="🔎 Starting scan...")

    # Run scan
    results, logs = scan_stocks(
        tickers=tickers,
        auto=(mode == "Auto Scan 100+"),
        update_progress=lambda p: progress_bar.progress(p, text=f"🔍 Scanning... {int(p * 100)}%"),
        st_log=log_area,
    )

    if not results:
        output_area.warning("⚠️ No valid breakout scores found or no data available.")
    else:
        try:
            df = pd.DataFrame(results)
            df = df.sort_values(by="Breakout Score", ascending=False)
            st.session_state["results"] = df
            output_area.dataframe(df, use_container_width=True)
        except Exception as e:
            output_area.error(f"❌ Error displaying results: {e}")

    # Show logs
    if logs:
        st.session_state["logs"] = logs
        with st.expander("📝 Scan Logs"):
            for log in logs:
                st.markdown(log)

# If no results yet
if "results" not in st.session_state:
    st.info("⚠️ No results found or scan hasn’t been run yet.")
