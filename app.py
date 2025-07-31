import streamlit as st
from scanner import scan_stocks

st.set_page_config(page_title="Money Maker", layout="centered")

st.title("💸 Money Maker – AI Stock Breakout Assistant")
st.markdown("Enter a list of stock tickers separated by commas (e.g. `AAPL, MSFT, TSLA`)")

tickers_input = st.text_input("Ticker Symbols", value="AAPL, MSFT, TSLA, PLTR, NVDA")

# ✅ Only define and use results when button is clicked
if st.button("🚀 Run Breakout Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("Please enter at least one valid ticker.")
    else:
        results = scan_stocks(tickers)

        st.markdown("---")
        for result in results:
            st.subheader(f"📊 {result['Ticker']}")
            
            if 'Error' in result.get('Sentiment', ''):
                st.error(f"❌ Could not retrieve data for {result['Ticker']}")
                continue

            st.markdown(f"- **Breakout Score:** `{result['Breakout Score']}`")
            st.markdown(f"- **Volume Ratio:** `{result['Volume Ratio']}`")
            st.markdown(f"- **Momentum:** `{result['Momentum']}`")
            st.markdown(f"- **RSI:** `{result['RSI']}`")
            st.markdown(f"- **Sentiment:** `{result['Sentiment']}`")

            target = result.get('Target Price', 'N/A')
            stop = result.get('Stop Loss', 'N/A')

            if isinstance(target, (int, float)):
                st.markdown(f"🟢 **Target Price (1M):** `${target:.2f}`")
            else:
                st.markdown("🟡 **Target Price (1M):** N/A")

            if isinstance(stop, (int, float)):
                st.markdown(f"🔴 **Stop Loss:** `${stop:.2f}`")
            else:
                st.markdown("🟡 **Stop Loss:** N/A")

            if result.get("Buy Signal"):
                st.success("🔥 **Buy Signal Triggered!**")

            st.markdown("---")
