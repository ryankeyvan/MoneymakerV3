import yfinance as yf
import pandas as pd
import streamlit as st

from utils.sentiment import get_sentiment_score
from ml_model import predict_breakout


def run_breakout_scan(ticker_list):
    results = []

    for ticker in ticker_list:
        st.write(f"⏳ Scanning {ticker}...")

        try:
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if data.empty:
                st.warning(f"⚠️ No data for {ticker}")
                continue

            recent_data = data.tail(14)

            # Get closing and volume series
            close_series = recent_data["Close"]
            volume_series = recent_data["Volume"]

            if close_series.isnull().any() or volume_series.isnull().any():
                st.write(f"⚠️ Missing data in {ticker}")
                continue

            if close_series.iloc[0] == 0 or volume_series.mean() == 0:
                st.write(f"⚠️ Invalid values in {ticker}")
                continue

            # Feature calculations
            volume_ratio = volume_series.iloc[-1] / volume_series.mean()
            price_momentum = close_series.iloc[-1] / close_series.iloc[0]
            pct_change = close_series.pct_change()
            rsi_numerator = pct_change.mean()
            rsi_denominator = pct_change.std()
            rsi = 100 - (100 / (1 + (rsi_numerator / rsi_denominator))) if rsi_denominator != 0 else 50

            sentiment = get_sentiment_score(ticker)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            last_close = float(close_series.iloc[-1])
            signal = "🔥 Buy" if breakout_score >= 0.7 else "🧪 Watch"
            target_price = round(last_close * 1.15, 2) if breakout_score >= 0.7 else "N/A"
            stop_loss = round(last_close * 0.93, 2) if breakout_score >= 0.7 else "N/A"

            st.markdown(f"""
                ### 📊 {ticker} — {signal}
                - **Breakout Score:** `{breakout_score:.2f}`
                - **Price:** `${last_close:.2f}`
                - **Target (1M):** `{target_price}`
                - **Stop Loss:** `{stop_loss}`
                - **Sentiment:** `{sentiment}`
                ---
            """)

            results.append({
                "Ticker": ticker,
                "Breakout Score": breakout_score,
                "Target Price": target_price,
                "Stop Loss": stop_loss,
                "Sentiment": sentiment,
                "Signal": signal
            })

        except Exception as e:
            st.write(f"❌ Error with {ticker}: {e}")

    return results
