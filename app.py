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
                if isinstance(result['Current Price'], (int, float)):
                    st.markdown(f"💰 **Current Price:** `${result['Current Price']:.2f}`")
                else:
                    st.markdown("💰 **Current Price:** N/A")

                st.markdown(f"- **Breakout Score:** `{result['Breakout Score']}`")
                st.markdown(f"- **Volume Ratio:** `{result['Volume Ratio']}`")
                st.markdown(f"- **Momentum:** `{result['Momentum']}`")
                st.markdown(f"- **RSI:** `{result['RSI']}`")
                st.markdown(f"- **Sentiment:** `{result['Sentiment']}`")

                if isinstance(result.get("Target Price"), (int, float)):
                    st.markdown(f"🟢 **Target Price (1M):** `${result['Target Price']:.2f}`")
                else:
                    st.markdown("🟡 **Target Price (1M):** N/A")

                if isinstance(result.get("Stop Loss"), (int, float)):
                    st.markdown(f"🔴 **Stop Loss:** `${result['Stop Loss']:.2f}`")
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

# Use breakout score to sort even if Target Price is missing
top_10 = sorted(results, key=lambda x: x.get("Breakout Score", 0), reverse=True)[:10]

if not top_10:
    st.warning("⚠️ No valid breakout stocks found at the moment. Try again later.")
else:
    st.success("✅ Showing Top 10 Stocks with Highest Breakout Score:")
    for result in top_10:
        st.subheader(f"📊 {result.get('Ticker', 'Unknown')}")

        price = result.get("Current Price")
        st.markdown(f"💰 **Current Price:** `${price:.2f}`" if isinstance(price, (int, float)) else "💰 **Current Price:** N/A")

        st.markdown(f"- **Breakout Score:** `{result.get('Breakout Score', 'N/A')}`")
        st.markdown(f"- **Volume Ratio:** `{result.get('Volume Ratio', 'N/A')}`")
        st.markdown(f"- **Momentum:** `{result.get('Momentum', 'N/A')}`")
        st.markdown(f"- **RSI:** `{result.get('RSI', 'N/A')}`")
        st.markdown(f"- **Sentiment:** `{result.get('Sentiment', 'N/A')}`")

        tp = result.get("Target Price")
        st.markdown(f"🟢 **Target Price (1M):** `${tp:.2f}`" if isinstance(tp, (int, float)) else "🟡 **Target Price (1M):** N/A")

        sl = result.get("Stop Loss")
        st.markdown(f"🔴 **Stop Loss:** `${sl:.2f}`" if isinstance(sl, (int, float)) else "🟡 **Stop Loss:** N/A")

        if result.get("Buy Signal"):
            st.success("🔥 **Buy Signal Triggered!**")

        st.markdown("---")

        st.success("✅ Showing Top 10 Stocks with Highest Breakout Score:")
        for result in top_10:
            st.subheader(f"📊 {result['Ticker']}")
            if isinstance(result['Current Price'], (int, float)):
                st.markdown(f"💰 **Current Price:** `${result['Current Price']:.2f}`")
            else:
                st.markdown("💰 **Current Price:** N/A")

            st.markdown(f"- **Breakout Score:** `{result['Breakout Score']}`")
            st.markdown(f"- **Volume Ratio:** `{result['Volume Ratio']}`")
            st.markdown(f"- **Momentum:** `{result['Momentum']}`")
            st.markdown(f"- **RSI:** `{result['RSI']}`")
            st.markdown(f"- **Sentiment:** `{result['Sentiment']}`")

            if isinstance(result.get("Target Price"), (int, float)):
                st.markdown(f"🟢 **Target Price (1M):** `${result['Target Price']:.2f}`")
            else:
                st.markdown("🟡 **Target Price (1M):** N/A")

            if isinstance(result.get("Stop Loss"), (int, float)):
                st.markdown(f"🔴 **Stop Loss:** `${result['Stop Loss']:.2f}`")
            else:
                st.markdown("🟡 **Stop Loss:** N/A")

            if result.get("Buy Signal"):
                st.success("🔥 **Buy Signal Triggered!**")

            st.markdown("---")
