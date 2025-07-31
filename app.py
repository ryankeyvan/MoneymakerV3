import streamlit as st
from scanner import scan_stocks
import pandas as pd

st.set_page_config(page_title="Money Maker – AI Stock Breakout Assistant", layout="centered")
st.title("💸 Money Maker – AI Stock Breakout Assistant")

# 📘 Indicator Key in Sidebar
with st.sidebar:
    st.header("📘 Indicator Key")
    st.markdown("""
    **RSI (Relative Strength Index)**  
    • <30 = Oversold 📉  
    • >70 = Overbought 📈  
    
    **Momentum**  
    • Positive = Uptrend 🟢  
    • Negative = Downtrend 🔴  
    
    **Volume Change %**  
    • >25% = Strong volume 🔥  
    • 10–25% = Moderate volume 📊  
    """)

# Choose Scan Mode
scan_mode = st.radio("Choose Scan Mode:", ["Manual Tickers", "Auto Scan ($5+)"])

results = []

# Manual Mode
if scan_mode == "Manual Tickers":
    tickers_input = st.text_input("Enter comma-separated tickers (e.g., AAPL,MSFT,TSLA)")
    if st.button("🔍 Run Manual Breakout Scan") and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        results = scan_stocks(tickers=tickers, auto=False)

# Auto Scan Mode
elif scan_mode == "Auto Scan ($5+)":
    st.subheader("🔎 Auto Scan – Top Stocks Over $5")
    if st.button("🚀 Run Auto Breakout Scan"):
        results = scan_stocks(tickers=None, auto=True)

# Display Results
if results:
    st.success(f"✅ {len(results)} stocks analyzed. Showing top 10 by breakout score.")
    df = pd.DataFrame(results)
    df = df.sort_values(by="Breakout Score", ascending=False).head(10)

    for _, row in df.iterrows():
        st.subheader(f"📈 {row['Ticker']} — {'🔥 Buy' if row['Breakout Score'] >= 0.7 else '👀 Watch'}")
        st.markdown(f"- **Current Price:** `${row['Current Price']}`")
        st.markdown(f"- **Breakout Score:** `{row['Breakout Score']}`")
        st.markdown(f"- **Target Price:** `${row['Target Price']}`")
        st.markdown(f"- **Stop Loss:** `${row['Stop Loss']}`")
        st.markdown(f"- **RSI:** `{row['RSI']}`")
        st.markdown(f"- **Momentum:** `{row['Momentum']}`")
        st.markdown(f"- **Volume Change:** `{row['Volume Change']}%`")
        st.markdown(f"- **Sentiment Score:** `{row['Sentiment Score']}`")
        st.markdown("---")

    # CSV Export
    st.download_button(
        label="⬇️ Export Results to CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="breakout_scan_results.csv",
        mime="text/csv"
    )

elif scan_mode and not results:
    st.warning("⚠️ No results found or scan hasn’t been run yet.")
