import requests

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
def generate_recommendations(sentiment_counts, avg_sentiment_score):
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
        # Use avg_sentiment_score to distinguish further
        if avg_sentiment_score < -0.6:
            condition = "depression"
        elif avg_sentiment_score < -0.2:
            condition = "anxiety"
        else:
            condition = "stress"
    else:
        condition = "stress"  # Default fallback

    # Step 3: Fetch recommendations
    recommendations = {
        "condition": condition,
        "music": [],
        "exercise": [],
        "articles": []
    }

    if condition in MUSIC_RECOMMENDATIONS:
        recommendations["music"] = fetch_youtube_videos(MUSIC_RECOMMENDATIONS[condition])

    if condition in EXERCISE_RECOMMENDATIONS:
        recommendations["exercise"] = fetch_youtube_videos(EXERCISE_RECOMMENDATIONS[condition])

    if condition in ARTICLE_RECOMMENDATIONS:
        recommendations["articles"] = fetch_articles(ARTICLE_RECOMMENDATIONS[condition])

    return recommendations
