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

            # ✅ Check that recent_data has valid volume and price
            if recent_data["Volume"].mean().item() == 0 or recent_data["Close"].iloc[0].item() == 0:
                st.write(f"⚠️ Invalid price or volume data for {ticker}")
                continue

            volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
            price_momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0]
            rsi_numerator = recent_data["Close"].pct_change().mean()
            rsi_denominator = recent_data["Close"].pct_change().std()
            rsi = 100 - (100 / (1 + (rsi_numerator / rsi_denominator))) if rsi_denominator != 0 else 50

            sentiment = get_sentiment_score(ticker)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            last_close = float(recent_data["Close"].iloc[-1])

            signal = "🔥 Buy" if breakout_score >= 0.7 else "🧪 Watch"
            target_price = round(last_close * 1.15, 2) if breakout_score >= 0.7 else "N/A"
            stop_loss = round(last_close * 0.93, 2) if breakout_score >= 0.7 else "N/A"

            st.write(f"{ticker}: Breakout Score={breakout_score:.2f}, Close=${last_close:.2f}, Sentiment={sentiment}")

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
