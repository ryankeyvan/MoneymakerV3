# Scan button
col1, col2 = st.columns([2, 1])
run_scan = col1.button("🚀 Run Scan")
test_scan = col2.button("🧪 Test AAPL")

results = []

if run_scan:
    tickers = []
    if auto_scan:
        tickers = get_all_stocks_above_5_dollars()
        st.info("🔍 Scanning top stocks...")
    elif input_tickers:
        tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
        st.info(f"🔍 Scanning: {', '.join(tickers)}")
    else:
        st.warning("⚠️ Add tickers or enable auto scan.")
        st.stop()

    # Progress bar
    progress = st.progress(0.0, text="Starting scan...")
    results = scan_stocks(tickers=tickers, update_progress=lambda p: progress.progress(p, text=f"Scanning... {int(p*100)}%"))
    progress.empty()

if test_scan:
    st.info("🧪 Testing scan on AAPL...")
    results = scan_stocks(tickers=["AAPL"])
