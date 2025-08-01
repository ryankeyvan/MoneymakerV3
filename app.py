import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scanner import scan_tickers, get_sp500_tickers, top_breakouts

st.set_page_config(page_title="MoneyMakerV3 AI Stock Breakout", layout="wide")
st.title("💰 MoneyMakerV3 AI Stock Breakout Assistant")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

#
# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
#
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

# Option to run a full S&P 500 scan
scan_sp500 = st.sidebar.checkbox("Scan entire S&P 500", value=False)
if scan_sp500:
    tickers = get_sp500_tickers()
    st.sidebar.info(f"Scanning {len(tickers)} tickers… this can take several minutes!")
else:
    use_wl = st.sidebar.checkbox("Use watchlist", value=True)
    if use_wl:
        tickers = st.session_state.watchlist
    else:
        raw = st.sidebar.text_input("Tickers to scan")
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

#
# ─── MAIN ────────────────────────────────────────────────────────────────────────
#
st.subheader("📈 Stock Scanner")
if st.button("Run Scan"):
    if not tickers:
        st.warning("No tickers provided. Add to watchlist, paste some, or check “Scan entire S&P 500.”")
    else:
        with st.spinner(f"Scanning {len(tickers)} tickers…"):
            results, failures = scan_tickers(tickers)

        df = pd.DataFrame(results)
        st.subheader("🔍 All Scan Results")
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"scan_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )

        if failures:
            st.subheader("⚠️ Failures")
            st.write(failures)

        # show top‐5 for each horizon
        for h, label in [('1w','1-Week'),('1m','1-Month'),('3m','3-Month')]:
            tops = top_breakouts(results, h, top_n=5)
            if tops:
                st.subheader(f"🔝 Top 5 {label} Breakouts")
                st.table(pd.DataFrame(tops)[[
                    'ticker','current_price', f'score_{h}', f'target_{h}'
                ]])
            else:
                st.info(f"No BUY signals for {label} horizon.")

        # price chart
        if not df.empty:
            st.subheader("📊 Price Chart")
            choice = st.selectbox("Select ticker to chart", df["ticker"].tolist())
            hist = yf.download(choice, period="6mo", interval="1d", progress=False)
            if not hist.empty:
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], linewidth=1.5)
                ax.set_title(f"{choice} – 6mo Close Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                st.pyplot(fig)
            else:
                st.info("No historical data for this ticker.")
        else:
            st.info("No successful scan results to chart.")
