import streamlit as st
from scanner import scan_stocks
import pandas as pd

st.set_page_config(page_title="Money Maker – AI Stock Breakout Assistant", layout="centered")

st.title("💸 Money Maker – AI Stock Breakout Assistant")

# Sidebar Legend
with st.sidebar:
    st.header("📘 Indicator Key")
    st.markdown("""
    **RSI (Relative Strength Index)**  
    • <30 = Oversold 📉  
    • >70 = Overbought 📈  
    
    **Momentum**  
    • Measures price acceleration  
    • Positive = Uptrend 🟢  
    • Negative = Downtrend 🔴  
    
    **Volume**  
    • Higher = More confirmation on breakout  
    """)

# Mode selection
scan_mode = st.radio("Choose Scan Mode:", ["Manual Tickers", "Auto Scan ($5+)"])

# Manual Ticker Input
if scan_mode == "Manual Tickers":
    tickers_input = st.text_input("Enter comma-separated tickers (e.g., AAPL,MSFT,TSLA)")
    if st.button("🔍 Run Manual Breakout Scan") and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        results = scan_stocks(tickers=tickers, auto=False)

# Auto Scan
else:
    st.markdown("## 🔎 Auto-Scan Mode – Top 10 Breakout Stocks > $5")
    if st.button("🚀 Run Auto Breakout Scan"):
        results = scan_stocks(tickers=None, auto=True)

# Display Results
if "results" in locals() and results:
    st.success("✅ Scan complete.")
    df = pd.DataFrame(results)

    for stock in results:
        st.markdown(f"### 📈 {stock['Ticker']} — {'🔥 Buy' if stock['Breakout Score'] > 0.7 else '⚠️ Watch'}")
        st.markdown(f"**Current Price:** ${stock['Price']:.2f}")
        st.markdown(f"**Breakout Score:** {stock['Breakout Score']:.2f}")
        st.markdown(f"**Target Price (1M):** {stock['Target Price'] if stock['Target Price'] else 'N/A'}")
        st.markdown(f"**Stop Loss:** {stock['Stop Loss'] if stock['Stop Loss'] else 'N/A'}")
        st.markdown(f"**Sentiment Score:** {stock['Sentiment'] if stock['Sentiment'] else 'N/A'}")
        st.markdown("---")

    # Download button
    st.download_button(
        label="⬇️ Export Results to CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="breakout_results.csv",
        mime="text/csv"
    )
elif "results" in locals():
    st.warning("⚠️ No breakout candidates found. Try again with different tickers or mode.")
