import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


reviews = [
    "I love this product! It's amazing!",
    "The quality is terrible, never buying again.",
    "It's okay, not great, not terrible.",
    "Outstanding service and support."
]


for review in reviews:
    sentiment_scores = sid.polarity_scores(review)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print(f"Compound Score: {compound_score}")
    print()
