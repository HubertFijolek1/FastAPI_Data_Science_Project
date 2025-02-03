import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_sentiment(text: str):
    """
    Returns sentiment scores using NLTK's VADER.
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)