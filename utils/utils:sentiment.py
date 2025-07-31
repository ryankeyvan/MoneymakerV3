# utils/sentiment.py

import random

def get_sentiment_score(ticker):
    """
    Simulates sentiment score from social media.
    Replace this with real Reddit/Twitter API later.
    """
    fake_scores = {
        "very positive": 3,
        "positive": 2,
        "neutral": 1,
        "negative": -1,
        "very negative": -2
    }

    # Randomly assign a fake sentiment score (simulate social chatter)
    score = random.choice(list(fake_scores.keys()))
    return score
