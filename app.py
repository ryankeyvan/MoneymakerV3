import streamlit as st
from scanner import scan_stocks, get_all_stocks_above_5_dollars

st.set_page_config(page_title="📈 Money Maker AI", layout="wide")

st.title("💸 Money Maker AI - Breakout Stock Scanner")

# Sidebar inputs
st.sidebar.header("📊 Scanner Options")
mode = st.sidebar.radio("Select Scan Mode", ["Enter Tickers", "Auto Scan 100+"])

if mode == "Enter Tickers":
    tickers_input = st.sidebar.text_area("Enter Tickers (comma separated)", "AAPL, TSLA, NVDA, MSFT")
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
else:
    tickers = get_all_stocks_above_5_dollars()

scan_button = st.sidebar.button("🚀 Run Scan")

# Display key/legend
with st.expander("📘 Indicator Key"):
    st.markdown("""
    - **Breakout Score**: Confidence (0–1) of breakout potential. ≥ 0.7 is a strong signal.
    - **RSI (Relative Strength Index)**: Measures overbought/oversold (30–70 ideal range).
    - **Momentum**: Price gain (%) over last 14 days.
    - **Volume Change**: Spike vs. 2-week average (%).
    - **Signal**: 🔥 = Strong Buy, 🧐 = Watch.
    """)

progress = st.empty()
log_area = st.empty()
output_area = st.empty()

if scan_button:
    st.session_state["results"] = []
    st.session_state["logs"] = []

    progress_bar = progress.progress(0, text="🔎 Starting scan...")

    results, logs = scan_stocks(
        tickers=tickers,
        update_progress=lambda p: progress_bar.progress(p, text=f"🔍 Scanning... {int(p * 100)}%"),
    )

    if not results:
        output_area.warning("⚠️ No valid breakout scores found. Some tickers may have failed to fetch data.")
    else:
        df = st.session_state["results"] = results
        df = df.sort_values(by="Breakout Score", ascending=False)
        output_area.dataframe(df)

    if logs:
        st.session_state["logs"] = logs
        with st.expander("📝 Scan Logs"):
            for log in logs:
                st.write(log)

# Default message
if "results" not in st.session_state:
    st.info("⚠️ No results found or scan hasn’t been run yet.")
