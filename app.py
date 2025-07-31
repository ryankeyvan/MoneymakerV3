import streamlit as st
import pandas as pd
from scanner import scan_stocks

# Optional import for auto mode
try:
    from scanner import get_all_stocks_above_5_dollars
    HAS_AUTO_SCAN = True
except ImportError:
    HAS_AUTO_SCAN = False

st.set_page_config(page_title="📈 Money Maker AI", layout="wide")
st.title("💸 Money Maker AI - Breakout Stock Scanner")

# Sidebar
st.sidebar.header("📊 Scanner Options")
mode_options = ["Enter Tickers"]
if HAS_AUTO_SCAN:
    mode_options.append("Auto Scan 100+")
mode = st.sidebar.radio("Select Scan Mode", mode_options)

if mode == "Enter Tickers" or not HAS_AUTO_SCAN:
    tickers_input = st.sidebar.text_area("Enter Tickers (comma separated)", "AAPL, TSLA, NVDA, MSFT")
    tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
else:
    tickers = get_all_stocks_above_5_dollars()

scan_button = st.sidebar.button("🚀 Run Scan")

# Indicator key
with st.expander("📘 Indicator Key"):
    st.markdown("""
    - **Breakout Score**: Confidence (0–1) of breakout potential. ≥ 0.7 = strong signal.
    - **RSI**: Relative Strength Index (30–70 = neutral; >70 = overbought).
    - **Momentum**: 14-day price change (%).
    - **Volume Change**: Volume spike vs. 14-day average (%).
    - **Signal**: 🔥 Buy / 🧐 Watch
    """)

progress = st.empty()
log_area = st.empty()
output_area = st.empty()

# Scan triggered
if scan_button:
    st.session_state["results"] = []
    st.session_state["logs"] = []

    progress_bar = progress.progress(0, text="🔎 Starting scan...")
    results, logs = scan_stocks(
        tickers=tickers,
        auto=(mode == "Auto Scan 100+"),
        update_progress=lambda p: progress_bar.progress(p, text=f"🔍 Scanning... {int(p * 100)}%"),
    )

    if not results or not isinstance(results, list) or len(results) == 0:
        output_area.warning("⚠️ No valid breakout scores found or no data available.")
    else:
        try:
            df = pd.DataFrame(results)
            df = df.sort_values(by="Breakout Score", ascending=False)
            st.session_state["results"] = df
            output_area.dataframe(df)
        except Exception as e:
            output_area.error(f"❌ Error converting results to table: {e}")

    if logs:
        st.session_state["logs"] = logs
        with st.expander("📝 Scan Logs"):
            for log in logs:
                st.write(log)

# Download Results
if "results" in st.session_state and not st.session_state["results"].empty:
    st.download_button(
        label="📥 Download Results as CSV",
        data=st.session_state["results"].to_csv(index=False),
        file_name="breakout_scan_results.csv",
        mime="text/csv",
    )

    if st.button("⭐ Save Watchlist"):
        st.session_state["results"].to_csv("watchlist.csv", index=False)
        st.success("✅ Watchlist saved as `watchlist.csv`")

    # Chart Viewer
    ticker_list = st.session_state["results"]["Ticker"].tolist()
    selected = st.selectbox("📊 View Chart for:", ticker_list)
    if selected:
        import yfinance as yf
        try:
            chart_data = yf.download(selected, period="1mo", interval="1d", progress=False)
            if not chart_data.empty:
                st.line_chart(chart_data["Close"])
        except Exception as e:
            st.warning(f"Could not load chart: {e}")

else:
    st.info("⚠️ No results found or scan hasn’t been run yet.")
