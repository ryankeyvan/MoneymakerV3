import streamlit as st
import pandas as pd
from scanner import scan_stocks

st.set_page_config(page_title="Money Maker AI", layout="wide")
st.title("💸 Money Maker: AI Stock Breakout Scanner")

# Sidebar controls
st.sidebar.header("🔧 Options")
input_tickers = st.sidebar.text_input("Enter ticker symbols (comma-separated)", "")
auto_scan = st.sidebar.checkbox("Auto-scan 100+ popular stocks over $5", value=False)

# Initialize results session
if "results" not in st.session_state:
    st.session_state["results"] = []

# LEGEND
st.markdown("""
### 📘 Indicator Key
- **Breakout Score**: AI confidence (0–1) of breakout in next 1–2 weeks  
- **RSI**: Relative Strength Index; >70 = overbought, <30 = oversold  
- **Momentum**: % change over 14 days  
- **Volume Change**: Compared to 14-day average  
- **Sentiment**: News sentiment from Yahoo headlines  
""")

# Run scan on button press
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Run Scan"):
        if auto_scan:
            st.info("🔎 Scanning 100+ popular tickers...")
            st.session_state["results"] = scan_stocks(auto=True)
        elif input_tickers:
            tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
            st.info(f"🔍 Scanning: {', '.join(tickers)}")
            st.session_state["results"] = scan_stocks(tickers=tickers)
        else:
            st.warning("⚠️ Please enter tickers or enable auto-scan.")

with col2:
    if st.button("🧪 Test Scan on AAPL Only"):
        st.session_state["results"] = scan_stocks(tickers=["AAPL"])

# Display results if available
results = st.session_state["results"]
if results:
    df = pd.DataFrame(results)
    if "Breakout Score" in df.columns:
        df = df.sort_values(by="Breakout Score", ascending=False).head(10)

    st.success(f"✅ Showing {len(df)} results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="breakout_results.csv", mime="text/csv")
else:
    st.warning("⚠️ No results found or scan hasn’t been run yet.")
