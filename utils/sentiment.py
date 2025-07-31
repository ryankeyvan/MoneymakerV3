# utils/sentiment.py
import yfinance as yf

positive_keywords = ["beat", "growth", "strong", "up", "record", "surge", "buy", "bullish", "outperform"]
negative_keywords = ["miss", "loss", "down", "warning", "lawsuit", "drop", "sell", "bearish", "underperform"]

def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        headlines = stock.news
        if not headlines:
            return 0.0

        score = 0
        for article in headlines[:10]:  # Analyze top 10 headlines
            title = article["title"].lower()
            for word in positive_keywords:
                if word in title:
                    score += 1
            for word in negative_keywords:
                if word in title:
                    score -= 1

        normalized_score = max(min(score / 10, 1), -1)  # Clamp between -1 and 1
        return round(normalized_score, 2)
    except Exception as e:
        print(f"Sentiment error for {ticker}: {e}")
        return 0.0
