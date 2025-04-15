from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import json
import firebase_admin
from firebase_admin import credentials, firestore
import re
import uuid
from recommendation_engine import generate_recommendations

from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


#app = Flask(__name__)
#CORS(app, origins=["http://172.20.128.1:3000"]) 

# Firebase setup
cred = credentials.Certificate(r"D:\\Downloads\\fyp1 final-20250414T201847Z-001\\fyp1 final\\realtimedb-8e3c1-firebase-adminsdk-fbsvc-3e649e0d98.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load chatbot model
model_path = r"D:\\Downloads\\fyp1 final-20250414T201847Z-001\\fyp1 final\\blenderbot_exact_mapping"
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load sentiment analysis model
sentiment_model_name = "cointegrated/rubert-tiny-sentiment-balanced"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Load exact context-response mapping
with open(r"D:\\Downloads\\fyp1 final-20250414T201847Z-001\\fyp1 final\\combined_file_normalized.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
context_response_map = {entry["Context"]: entry["Response"] for entry in data}

# Session state
conversation_id = str(uuid.uuid4())
conversation_log = []
sentiment_scores = []
sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
sensitive_words = ["kill", "rape", "sex", "shoot", "kidnap", "murder", "attack"]

def get_sentiment_score(label):
    return {"positive": 1, "neutral": 0, "negative": -1}.get(label, 0)

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]["label"]

def chatbot_response(input_text):
    if any(word in input_text.lower() for word in sensitive_words):
        return "I think you are thinking too much, take a break and relax for a while.", "neutral"

    greetings = ["hello", "hi", "hey", "hola", "howdy", "helo", "yo"]
    slang_responses = {
        "sup": "Not much! How about you?",
        "yo": "Hey there!",
        "wassup": "Just here to chat! How about you?",
        "bruh": "What's up?",
        "lmao": "Glad to hear you're laughing!",
        "omg": "What happened?",
        "idk": "It's okay to not know everything!",
        "im sad": "Please don't be sad! Share with me."
    }

    if input_text.lower() in greetings:
        return "Hello! How can I help you today?", "neutral"
    if input_text.lower() in slang_responses:
        return slang_responses[input_text.lower()], "neutral"
    if re.fullmatch(r"[0-9\W]+", input_text) or re.fullmatch(r"(.)\1{2,}", input_text) or re.fullmatch(r"[a-zA-Z]{10,}", input_text):
        return "Sorry, I don't understand that.", "neutral"

    sentiment = analyze_sentiment(input_text)
    sentiment_score = get_sentiment_score(sentiment)
    sentiment_scores.append(sentiment_score)
    sentiment_counts[sentiment] += 1

    if input_text in context_response_map:
        response = context_response_map[input_text]
    else:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
        outputs = chatbot_model.generate(inputs["input_ids"], max_length=128, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    conversation_log.append({"user": input_text, "sentiment": sentiment, "bot": response})
    return response, sentiment

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_log, sentiment_scores, sentiment_counts

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    if user_input.lower() in ["exit", "bye", "quit"]:
        sentiment = analyze_sentiment(user_input)
        sentiment_scores.append(get_sentiment_score(sentiment))
        sentiment_counts[sentiment] += 1

        response = "Goodbye! Have a nice day 😊"
        conversation_log.append({
            "user": user_input,
            "sentiment": sentiment,
            "bot": response
        })

        return jsonify({
            "response": response,
            "sentiment": sentiment,
            "end_chat": True
        })

    # For non-exit messages
    bot_response, sentiment = chatbot_response(user_input)
    
    # Update sentiment counts for both user and bot messages
    user_sentiment = analyze_sentiment(user_input)
    bot_sentiment = sentiment  # Already returned by chatbot_response

    sentiment_scores.append(get_sentiment_score(user_sentiment))
    sentiment_counts[user_sentiment] += 1

    sentiment_scores.append(get_sentiment_score(bot_sentiment))
    sentiment_counts[bot_sentiment] += 1

    # Add both user and bot conversation to the log
    conversation_log.append({
        "role": "user",
        "message": user_input,
        "sentiment": user_sentiment,
        "score": get_sentiment_score(user_sentiment)
    })
    
    conversation_log.append({
        "role": "bot",
        "message": bot_response,
        "sentiment": bot_sentiment,
        "score": get_sentiment_score(bot_sentiment)
    })

    return jsonify({
        "response": bot_response,
        "sentiment": bot_sentiment,
        "end_chat": False
    })

@app.route("/end-session", methods=["POST"])
def end_session():
    global conversation_id, conversation_log, sentiment_scores, sentiment_counts

    # Calculate average sentiment score
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Save the conversation to Firestore
    db.collection("chatbot_conversations").document(conversation_id).set({
        "conversation_id": conversation_id,
        "messages": conversation_log,
        "sentiment_counts": sentiment_counts,
        "average_sentiment_score": average_sentiment
    })

    # Filter keys to ensure only expected ones are passed
    filtered_sentiment_counts = {
        k: sentiment_counts.get(k, 0) for k in ["positive", "neutral", "negative"]
    }

    # Generate full recommendation object
    rec_obj = generate_recommendations(filtered_sentiment_counts, average_sentiment)

    # Flatten links
    all_recommendations = rec_obj["music"] + rec_obj["exercise"] + rec_obj["articles"]

    result = {
        "message": "Conversation saved.",
        "sentiment_counts": sentiment_counts,
        "average_sentiment_score": average_sentiment,
        "recommendations": all_recommendations,
        "condition": rec_obj["condition"]
    }

    # Reset session variables
    conversation_id = str(uuid.uuid4())
    conversation_log = []
    sentiment_scores = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    return jsonify(result), 200

@app.route("/generate-recommendations", methods=["POST"])
def generate_recommendations_route():
    data = request.get_json()

    # Prefer this format
    sentiment_counts = data.get("sentiment_counts")
    average_sentiment = data.get("average_sentiment_score")

    # Fallback if needed
    if not sentiment_counts and "report" in data:
        report = data["report"]
        sentiment_counts = {
            "positive": report.get("Positive", 0),
            "neutral": report.get("Neutral", 0),
            "negative": report.get("Negative", 0),
        }
        average_sentiment = 0.0  # or calculate from counts if needed

    if not sentiment_counts:
        return jsonify({"error": "Missing sentiment data"}), 400

    # Generate full recommendation object
    rec_obj = generate_recommendations(sentiment_counts, average_sentiment)

    # Flatten the links
    all_recommendations = rec_obj["music"] + rec_obj["exercise"] + rec_obj["articles"]

    return jsonify({
        "recommendations": all_recommendations,
        "condition": rec_obj["condition"]
    }), 200

@app.route("/get-report", methods=["GET"])
def get_report():
    report_id = request.args.get("id")
    if not report_id:
        return jsonify({"error": "Missing conversation ID"}), 400

    doc_ref = db.collection("chatbot_conversations").document(report_id)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Report not found"}), 404

    data = doc.to_dict()
    sentiment_counts = data.get("sentiment_counts", {})
    average_sentiment = data.get("average_sentiment_score", 0.0)

    # Generate recommendations again (in case they weren't stored)
    rec_obj = generate_recommendations(sentiment_counts, average_sentiment)
    all_recommendations = rec_obj["music"] + rec_obj["exercise"] + rec_obj["articles"]

    return jsonify({
        "conversation_id": report_id,
        "messages": data.get("messages", []),
        "sentiment_counts": sentiment_counts,
        "average_sentiment_score": average_sentiment,
        "recommendations": all_recommendations,
        "condition": rec_obj["condition"]
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
