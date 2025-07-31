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

            # Guard against zero division and NaN
            if recent_data["Volume"].mean() == 0 or recent_data["Close"].iloc[0] == 0:
                st.write(f"⚠️ Invalid price or volume data for {ticker}")
                continue

            volume_ratio = recent_data["Volume"].iloc[-1] / recent_data["Volume"].mean()
            price_momentum = recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[0]
            rsi = 100 - (100 / (1 + (recent_data["Close"].pct_change().mean() / recent_data["Close"].pct_change().std())))

            sentiment = get_sentiment_score(ticker)
            breakout_score = predict_breakout(volume_ratio, price_momentum, rsi)

            if breakout_score > 0.7:
                last_close = float(recent_data["Close"].iloc[-1])
                target_price = round(last_close * 1.15, 2)
                stop_loss = round(last_close * 0.93, 2)

                # ✅ Bonus debug log
                st.write(f"✅ {ticker} breakout score: {breakout_score:.2f}, last close: ${last_close:.2f}")

                results.append({
                    "Ticker": ticker,
                    "Breakout Score": breakout_score,
                    "Target Price": target_price,
                    "Stop Loss": stop_loss,
                    "Sentiment": sentiment,
                    "Signal": "🔥 Buy"
                })

        except Exception as e:
            st.write(f"❌ Error with {ticker}: {e}")

    return results
