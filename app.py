# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scanner import scan_tickers

st.set_page_config(page_title="MoneyMakerV3 AI Stock Breakout", layout="wide")
st.title("💰 MoneyMakerV3 AI Stock Breakout Assistant")

# persist watchlist
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# sidebar
st.sidebar.header("🍽️ Watchlist")
inp = st.sidebar.text_input("Add tickers (comma separated)")
if st.sidebar.button("Add"):
    new = [t.strip().upper() for t in inp.split(",") if t.strip()]
    st.session_state.watchlist = list(dict.fromkeys(st.session_state.watchlist + new))
    st.sidebar.success(f"Added: {', '.join(new)}")
if st.sidebar.button("Clear"):
    st.session_state.watchlist = []
    st.sidebar.info("Watchlist cleared.")

st.sidebar.write("**Current:**", st.session_state.watchlist)

# scanner UI
st.subheader("📈 Stock Scanner")
use_wl = st.checkbox("Use watchlist", value=True)
if use_wl:
    tickers = st.session_state.watchlist
else:
    tickers = [t.strip().upper() for t in st.text_input("Tickers to scan").split(",") if t.strip()]

if st.button("Run Scan"):
    if not tickers:
        st.warning("No tickers provided.")
    else:
        with st.spinner("Scanning…"):
            results, failures = scan_tickers(tickers)
        df = pd.DataFrame(results)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv,
                           file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        if failures:
            st.subheader("⚠️ Failures")
            st.write(failures)

        st.subheader("📊 Price Chart")
        choice = st.selectbox("Select ticker", df["ticker"].tolist())
        if choice:
            hist = yf.download(choice, period="6mo", interval="1d", progress=False)
            if not hist.empty:
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], linewidth=1.5)
                ax.set_title(f"{choice} – 6mo Close Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                st.pyplot(fig)
            else:
                st.info("No history for this ticker.")
