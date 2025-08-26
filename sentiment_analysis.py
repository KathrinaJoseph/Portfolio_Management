import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

NEWS_API_KEY = "ebabc6ffb4fa4b99a3b1bd3f3252fe4c"

def analyze_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    scores = []

    for article in response.get("articles", []):
        score = analyzer.polarity_scores(article['title'])['compound']
        scores.append(score)

    if scores:
        return sum(scores) / len(scores)
    return 0