import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os

# API Keys
YOUTUBE_API_KEY = "AIzaSyAXeIl50ORTg9nLhpOUV0GsNxeWxsvN3gg"
GOOGLE_API_KEY = "AIzaSyBg7ClmibKPvjp0YF3cAtnitVu3priNDRQ"
SEARCH_ENGINE_ID = "455d1dcceb9074cf0"

# Recommendation Queries
MUSIC_RECOMMENDATIONS = {
    "stress": "Relaxing music for stress relief with soft piano, nature sounds, and guided meditation",
    "anxiety": "Guided meditation music for anxiety relief and slow breathing exercises",
    "normal": "Positive energy music for happiness and relaxation"
}

EXERCISE_RECOMMENDATIONS = {
    "stress": "5 to 10 breathing exercises for stress relief",
    "anxiety": "5 to 10 breathing exercises for anxiety",
    "depression": "yoga exercises for depression",
    "normal": "daily fitness routine"
}

ARTICLE_RECOMMENDATIONS = {
    "stress": "best articles on stress management",
    "anxiety": "how to cope with anxiety articles",
    "depression": "mental health blogs on depression",
    "normal": "positive and self-care blogs"
}

# Load professionals dataset
try:
    csv_path = os.path.join(os.path.dirname(__file__), "Final professionals dataset.csv")
    professionals_df = pd.read_csv(csv_path)
    professionals_df['combined_text'] = professionals_df.apply(
        lambda x: f"{x['Name']} {x['Profession']} {x['Experience']} {x['Location']} {x['Education']}",
        axis=1
    )
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(professionals_df['combined_text'])
    print(f"Professionals loaded: {professionals_df.shape}")
    print(f"Sample professional: {professionals_df.iloc[0].to_dict()}")
except Exception as e:
    print(f"Error loading professionals dataset: {e}")
    professionals_df = None
    tfidf_matrix = None

# Add a global index to rotate professionals
_prof_index = 0

def get_recommended_professionals():
    global _prof_index
    if professionals_df is None:
        return []
    n = len(professionals_df)
    if n == 0:
        return []
    # Get 3 professionals, rotating through the list
    start = _prof_index % n
    indices = [(start + i) % n for i in range(3)]
    _prof_index = (_prof_index + 3) % n
    def safe_get(prof, col):
        return prof[col] if col in prof and pd.notnull(prof[col]) else 'N/A'
    professionals = []
    for idx in indices:
        professional = professionals_df.iloc[idx]
        professionals.append({
            'name': safe_get(professional, 'Name'),
            'profession': safe_get(professional, 'Profession'),
            'experience': safe_get(professional, 'Experience'),
            'education': safe_get(professional, 'Education'),
            'location': safe_get(professional, 'Location'),
            'contact': safe_get(professional, 'To book appointments visit')
        })
    return professionals

# Internet Check
def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

# Fetch YouTube Videos
def fetch_youtube_videos(query):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&maxResults=5&type=video&key={YOUTUBE_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Check if 'items' exists and is a list
        if 'items' not in data or not isinstance(data['items'], list):
            print("⚠️ No items found in YouTube response")
            return [{"title": "No videos found", "url": "#"}]

        videos = []
        for item in data.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            })

        return videos if videos else [{"title": "No videos found", "url": "#"}]

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching YouTube videos: {e}")
        return [{"title": "Error retrieving videos", "url": "#"}]

# Fetch Articles
def fetch_articles(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'items' not in data or not isinstance(data['items'], list):
            print("⚠️ No items found in articles response")
            return [{"title": "No articles found", "link": "#"}]

        articles = []
        for item in data.get("items", []):
            articles.append({"title": item["title"], "link": item["link"]})

        return articles[:5] if articles else [{"title": "No articles found", "link": "#"}]

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching articles: {e}")
        return [{"title": "Error retrieving articles", "link": "#"}]

def map_sentiment_to_condition(sentiment_counts):
    positive = sentiment_counts.get("positive", 0)
    neutral = sentiment_counts.get("neutral", 0)
    negative = sentiment_counts.get("negative", 0)

    if positive > max(neutral, negative):
        return 'normal'
    elif negative > max(positive, neutral):
        return 'anxiety'  # or 'stress' or 'depression' depending on your design
    else:
        return 'stress'  # neutral or mixed case

# ✅ Main Recommendation Function for Flask Integration
def generate_recommendations(sentiment_counts, avg_sentiment_score, phq9_scores=None, gad7_scores=None):
    if not check_internet():
        return {"error": "No internet connection"}

    # Step 1: Determine dominant sentiment
    dominant = max(sentiment_counts, key=sentiment_counts.get)

    # Step 2: Map dominant sentiment to mental health condition
    if all(v == 0 for v in sentiment_counts.values()) or dominant == "neutral":
        condition = "normal"
    elif dominant == "positive":
        condition = "normal"
    elif dominant == "negative":
        if avg_sentiment_score < -0.6:
            condition = "depression"
        elif avg_sentiment_score < -0.2:
            condition = "anxiety"
        else:
            condition = "stress"
    else:
        condition = "stress"

    recommendations = {
        "condition": condition,
        "music": [],
        "exercise": [],
        "articles": [],
        "professionals": get_recommended_professionals()  # Always get professionals
    }

    # Always fetch categories for the detected condition, fallback to 'normal' if needed
    music_query = MUSIC_RECOMMENDATIONS.get(condition) or MUSIC_RECOMMENDATIONS.get('normal')
    exercise_query = EXERCISE_RECOMMENDATIONS.get(condition) or EXERCISE_RECOMMENDATIONS.get('normal')
    article_query = ARTICLE_RECOMMENDATIONS.get(condition) or ARTICLE_RECOMMENDATIONS.get('normal')

    recommendations["music"] = fetch_youtube_videos(music_query)
    if not recommendations["music"] or recommendations["music"][0].get('url') == '#':
        recommendations["music"] = [{"title": "Relaxing Piano Music", "url": "https://www.youtube.com/watch?v=1ZYbU82GVz4", "type": "music"}]

    recommendations["exercise"] = fetch_youtube_videos(exercise_query)
    if not recommendations["exercise"] or recommendations["exercise"][0].get('url') == '#':
        recommendations["exercise"] = [{"title": "Breathing Exercise for Stress Relief", "url": "https://www.youtube.com/watch?v=nmFUDkj1Aq0", "type": "exercise"}]

    recommendations["articles"] = fetch_articles(article_query)
    if not recommendations["articles"] or recommendations["articles"][0].get('link') == '#':
        recommendations["articles"] = [{"title": "How to Manage Stress", "link": "https://www.helpguide.org/articles/stress/stress-management.htm", "type": "article"}]

    # Add type field to each recommendation
    for rec in recommendations["music"]:
        rec["type"] = "music"
    for rec in recommendations["exercise"]:
        rec["type"] = "exercise"
    for rec in recommendations["articles"]:
        rec["type"] = "article"

    return recommendations
