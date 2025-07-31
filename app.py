import streamlit as st
from scanner import scan_stocks

# Page config
st.set_page_config(page_title="Money Maker", layout="centered")

# 📘 Sidebar: Metric Key / Legend
with st.sidebar:
    st.title("📖 Legend / Metric Key")
    st.markdown("""
    **📈 RSI (Relative Strength Index):**
    - `> 70` → Overbought  
    - `< 30` → Oversold  
    - `50–65` → Healthy uptrend  

    **⚡ Momentum (Price Trend):**
    - `> 1.05` → Strong upward momentum  
    - `< 1.00` → Weak or declining  

    **📊 Volume Ratio:**
    - `> 1.5` → High interest (potential breakout)  
    - `≈ 1.0` → Normal volume  
    - `< 0.8` → Low volume  
    """)
    st.markdown("---")
    st.info("✅ A '🔥 Buy Signal' means the model scored the stock **≥ 0.7** for breakout potential.")

# 📌 Mode Selection
mode = st.radio("Choose Scan Mode:", ["Manual Tickers", "Auto Scan ($5+)"])

# 🔹 MANUAL TICKER MODE
if mode == "Manual Tickers":
    st.title("💸 Money Maker – Manual Mode")
    st.markdown("Enter a list of stock tickers separated by commas (e.g. `AAPL, MSFT, TSLA`)")
    tickers_input = st.text_input("Ticker Symbols", value="AAPL, MSFT, TSLA, PLTR, NVDA")

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

# 🔹 AUTO MODE
elif mode == "Auto Scan ($5+)":
    st.title("🔍 Auto-Scan Mode – Top 10 Breakout Stocks > $5")
    if st.button("🚀 Run Auto Breakout Scan"):
        watchlist = [
            "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META", "AMD", "NFLX", "PLTR",
            "CRM", "SNOW", "SHOP", "INTC", "QCOM", "AVGO", "BABA", "DIS", "ABNB", "ROKU",
            "F", "GM", "UBER", "LYFT", "NKE", "WMT", "TGT", "XOM", "CVX", "MARA", "RIOT",
            "SOFI", "DKNG", "PYPL", "SQ", "COIN", "LULU", "SPOT", "ORCL", "CSCO", "V", "MA",
            "BA", "CCL", "DAL", "UAL", "BIDU", "JD", "PDD", "TSM", "ADBE", "Z", "ZIM", "TTD"
        ]

        results = scan_stocks(watchlist)
        filtered = [r for r in results if isinstance(r.get("Target Price"), (int, float)) and r.get("Target Price", 0) > 5]
        top_10 = sorted(filtered, key=lambda x: x["Breakout Score"], reverse=True)[:10]

        st.success("✅ Showing Top 10 Stocks with Highest Breakout Score:")
        for result in top_10:
            st.subheader(f"📊 {result['Ticker']}")
            st.markdown(f"- **Breakout Score:** `{result['Breakout Score']}`")
            st.markdown(f"- **Volume Ratio:** `{result['Volume Ratio']}`")
            st.markdown(f"- **Momentum:** `{result['Momentum']}`")
            st.markdown(f"- **RSI:** `{result['RSI']}`")
            st.markdown(f"- **Sentiment:** `{result['Sentiment']}`")
            st.markdown(f"🟢 **Target Price (1M):** `${result['Target Price']:.2f}`")
            st.markdown(f"🔴 **Stop Loss:** `${result['Stop Loss']:.2f}`")

            if result.get("Buy Signal"):
                st.success("🔥 **Buy Signal Triggered!**")
            st.markdown("---")
