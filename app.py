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

    # Safely check Target Price and Stop Loss
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
