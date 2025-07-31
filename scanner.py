import yfinance as yf
from ta.momentum import RSIIndicator

def scan_single_stock(ticker, ml_model=None):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            return None

        # Calculate technical indicators
        rsi = RSIIndicator(df["Close"]).rsi()
        momentum = df["Close"].diff(14)
        volume_change = df["Volume"].pct_change(periods=14) * 100

        rsi_val = rsi.iloc[-1] if not rsi.empty else 0
        momentum_val = momentum.iloc[-1] if not momentum.empty else 0
        volume_val = volume_change.iloc[-1] if not volume_change.empty else 0
        current_price = df["Close"].iloc[-1] if not df.empty else 0

        # Calculate breakout score
        score = 0
        if 45 < rsi_val < 65:
            score += 0.3
        if momentum_val > 0:
            score += 0.3
        if volume_val > 5:
            score += 0.4

        # Simple 1-month projected price estimation
        projected_price = current_price * (1 + score)

        return {
            "Ticker": ticker,
            "Breakout Score": round(score, 2),
            "RSI": round(rsi_val, 2),
            "Momentum": round(momentum_val, 2),
            "Volume Change": round(volume_val, 2),
            "Current Price": round(current_price, 2),
            "Target Price (1M)": round(projected_price, 2)
        }

    except Exception as e:
        return None
