# utils/charting.py

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_stock_chart(ticker):
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data["Close"], label="Close Price", color="blue")
    ax.set_title(f"{ticker} - Last 3 Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    return fig
