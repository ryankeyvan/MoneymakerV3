import streamlit as st
import pandas as pd
from scanner import scan_stocks, get_all_stocks_above_5_dollars

st.set_page_config(page_title="Money Maker AI", layout="wide")
st.title("💸 Money Maker: AI Stock Breakout Assistant")

# Sidebar
st.sidebar.header("🔧 Options")
input_tickers = st.sidebar.text_input("Enter ticker symbols (comma-separated)", "")
auto_scan = st.sidebar.checkbox("Auto-scan 100+ popular stocks over $5", value=False)

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = []

# Indicator key
with st.expander("📘 Indicator Key"):
    st.markdown("""
- **Breakout Score**: AI confidence (0–1) of breakout potential  
- **RSI**: Relative Strength Index; >70 = overbought, <30 = oversold  
- **Momentum**: % price change over 14 days  
- **Volume Change**: % above 14-day volume avg  
- **Sentiment Score**: News & social buzz score  
- **Target Price**: Projected +15%  
- **Stop Loss**: Suggested -7% downside buffer
""")

# Buttons
run_scan = st.button("🚀 Run Scan")
test_scan = st.button("🧪 Test on AAPL")

# Output container
log_area = st.container()

# Run scan
if run_scan:
    if auto_scan:
        tickers = get_all_stocks_above_5_dollars()
        st.info(f"🔍 Auto-scanning {len(tickers)} top stocks...")
    elif input_tickers:
        tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
        st.info(f"🔍 Scanning: {', '.join(tickers)}")
    else:
        st.warning("⚠️ Please enter tickers or enable auto-scan.")
        st.stop()

    progress = st.progress(0.0, text="⏳ Starting scan...")
    st.session_state["results"] = scan_stocks(
        tickers=tickers,
        update_progress=lambda p: progress.progress(p, text=f"🔎 Scanning... {int(p * 100)}%"),
        st_log=log_area
    )
    progress.empty()

if test_scan:
    st.info("🧪 Scanning AAPL...")
    st.session_state["results"] = scan_stocks(tickers=["AAPL"], st_log=log_area)

# Display results
results = st.session_state["results"]
if results:
    df = pd.DataFrame(results)
    if "Breakout Score" in df.columns:
        df = df.sort_values(by="Breakout Score", ascending=False)

    st.success(f"✅ Showing {len(df)} results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="breakout_results.csv", mime="text/csv")
else:
    st.warning("⚠️ No results found or scan hasn’t been run yet.")
