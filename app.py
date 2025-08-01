# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from scanner import scan_tickers, top_breakouts
from datetime import datetime

st.set_page_config(page_title="MoneyMakerV3", layout="wide")
st.title("💰 MoneyMakerV3 AI Stock Breakout Assistant")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔍 Scan S&P 500"):
        st.session_state['tickers'] = None
    st.markdown("---")
    txt = st.text_area("Or enter tickers (comma-sep):",
                       value=",".join(st.session_state.get('tickers',[])) if st.session_state.get('tickers') else "")
    if st.button("Set Tickers"):
        st.session_state['tickers'] = [t.strip().upper() for t in txt.split(",") if t.strip()]
    st.markdown("---")
    if st.button("Clear Results"):
        st.session_state.pop('results', None)

# Main
if 'results' not in st.session_state:
    st.info("Choose Scan S&P 500 or Set Tickers to begin.")
else:
    results = st.session_state['results']
    failures = st.session_state.get('failures', [])
    horizons = [('1w','1-Week'),('1m','1-Month'),('3m','3-Month')]

    tabs = st.tabs([lab for _,lab in horizons])
    for (h,k), tab in zip(horizons, tabs):
        with tab:
            st.subheader(f"Top 5 Breakouts ({k})")
            top5 = top_breakouts(results, h)
            if top5:
                df = pd.DataFrame(top5)
                df = df[['ticker','current_price', f'score_{h}', f'target_{h}']]
                df.columns = ['Ticker','Price', 'Score', 'Target']
                st.dataframe(df, use_container_width=True)
                # Show metrics
                cols = st.columns(len(df))
                for c,row in zip(cols, top5):
                    c.metric(label=row['ticker'],
                             value=f"${row['current_price']}",
                             delta=f"{int((row[f'target_{h}']/row['current_price']-1)*100)}%")
            else:
                st.write("No BUY signals this period.")

            # Chart selector
            ticker = st.selectbox(f"Chart ({k})", [r['ticker'] for r in results], key=h)
            hist = yf.download(ticker, period="6mo", interval="1d", progress=False)
            st.line_chart(hist['Close'], width=0, height=300)

    # Download full results
    df_all = pd.DataFrame(results)
    st.download_button("📥 Download Full CSV",
                       df_all.to_csv(index=False).encode(),
                       file_name=f"breakout_scan_{datetime.now():%Y%m%d_%H%M%S}.csv")

    if failures:
        st.error(f"⚠️ {len(failures)} failures; see sidebar.")
        st.sidebar.subheader("Fetch Failures")
        st.sidebar.write(failures)

# Run scan when needed
if (st.sidebar.button("Run Scan") and
    'tickers' in st.session_state):
    with st.spinner("Scanning… (this may take a few minutes)"):
        res, errs = scan_tickers(st.session_state['tickers'])
    st.session_state['results']  = res
    st.session_state['failures'] = errs
    st.experimental_rerun()
