import streamlit as st
from scanner import scan_stocks

# Streamlit App Config
st.set_page_config(page_title="Money Maker", layout="centered")

st.title("💸 Money Maker – AI Stock Breakout Assistant")
st.markdown("Enter a list of stock tickers separated by commas (e.g. `AAPL, MSFT, TSLA`)")

# Ticker input
tickers_input = st.text_input("Ticker Symbols", value="AAPL, MSFT, TSLA, PLTR, NVDA")

# Run button
if st.button("🚀 Run Breakout Scan"):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("Please enter at least one valid ticker.")
    else:
        results = scan_stocks(tickers)
        
        st.markdown("---")
        for result in results:
            if "Error" in result:
                st.error(f"{result['Ticker']}: {result['Error']}")
                continue

            st.subheader(f"📈 {result['Ticker']}")
            st.markdown(f"- **Breakout Score:** `{result['Breakout Score']}`")
            st.markdown(f"- **Target Price (1M):** `{result['Target Price']}`")
            st.markdown(f"- **Stop Loss:** `{result['Stop Loss']}`")
            st.markdown(f"- **Sentiment:** `{result['Sentiment']}`")
            st.markdown("---")
