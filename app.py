import streamlit as st
import pandas as pd
from scanner import scan_stocks

st.set_page_config(page_title="Money Maker AI", layout="wide")
st.markdown("<h1 style='text-align: center;'>💸 Money Maker AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Stock Breakout Scanner Powered by AI & Sentiment</h4>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    input_tickers = st.text_input("Manual Tickers (comma-separated)", "")
    auto_scan = st.checkbox("Scan Top 100 Stocks", value=False)
    st.markdown("### 🧠 Indicator Key")
    st.markdown("""
    - **Breakout Score**: AI prediction of breakout (0-1 scale)  
    - **RSI**: Overbought >70, Oversold <30  
    - **Momentum**: % price gain in 14 days  
    - **Volume Change**: vs 14-day avg  
    - **Sentiment**: News tone (0–1 scale)
    """)

# Scan button
col1, col2 = st.columns([2,1])
run_scan = col1.button("🚀 Run Scan")
test_scan = col2.button("🧪 Test AAPL")

# Results
results = []

if run_scan:
    if auto_scan:
        st.info("🔍 Scanning 100+ top stocks...")
        results = scan_stocks(auto=True)
    elif input_tickers:
        tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
        st.info(f"🔍 Scanning: {', '.join(tickers)}")
        results = scan_stocks(tickers=tickers)
    else:
        st.warning("⚠️ Add tickers or enable auto scan.")

if test_scan:
    st.info("🧪 Testing scan on AAPL...")
    results = scan_stocks(tickers=["AAPL"])

# Display results
if results:
    st.success(f"✅ Found {len(results)} stocks with breakout data")
    df = pd.DataFrame(results).sort_values("Breakout Score", ascending=False).head(10)

    for i, row in df.iterrows():
        with st.expander(f"📈 {row['Ticker']} | Score: {row['Breakout Score']} | Price: ${row['Current Price']}"):
            st.markdown(f"""
            - **Current Price**: ${row['Current Price']}
            - **Breakout Score**: `{row['Breakout Score']}`
            - **Target Price**: `${row['Target Price']}`
            - **Stop Loss**: `${row['Stop Loss']}`
            - **RSI**: `{row['RSI']}`
            - **Momentum**: `{row['Momentum']}%`
            - **Volume Change**: `{row['Volume Change']}%`
            - **Sentiment Score**: `{row['Sentiment Score']}`
            """)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv, file_name="breakout_results.csv", mime="text/csv")
else:
    st.info("📊 Results will show here after scan")
