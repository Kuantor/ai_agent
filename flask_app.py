"""
Standalone Flask web app for Tynna, the AI study assistant.

The chatbot logic lives in agent.py (`TynnaAgent`); this file is only the web
front-end. The KuantorFlow project imports the same `TynnaAgent`, so the agent
code is never duplicated.

Run:  python flask_app.py     (needs ANTHROPIC_API_KEY in .env)
"""

import anthropic
from flask import Flask, jsonify, render_template, request

from agent import TynnaAgent, api_error_response
from word_list import WordListGenerator
from cards_db import FlashcardsDB

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Load the knowledge base and Anthropic client once, reuse across requests.
agent = TynnaAgent()
word_list_gen = WordListGenerator(agent.kb.chunks)
cards_db = FlashcardsDB()


@app.route("/")
def home():
    """Render the main chat page."""
    return render_template("index.html", kb_chunks_count=agent.chunk_count)


@app.route("/about")
def about():
    """Render Tynna's profile and gallery page."""
    return render_template("about.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Answer a question via the shared TynnaAgent. Returns response + sources."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    history = data.get("history", [])
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        return jsonify(agent.answer(question, history))
    except anthropic.APIError as e:
        body, status = api_error_response(e)
        return jsonify(body), status
    except Exception:
        app.logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "Internal server error. Please try again later."}), 500


@app.route("/api/word-list", methods=["POST"])
def get_word_list():
    """Get a vocabulary word list from the knowledge base."""
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip() or None
    
    try:
        terms = word_list_gen.extract_key_terms(limit=15)
        formatted = word_list_gen.generate_word_list_with_translations(topic)
        return jsonify({
            "word_list": formatted,
            "terms": terms,
            "count": len(terms)
        })
    except Exception as e:
        app.logger.exception("Error generating word list")
        return jsonify({"error": "Failed to generate word list"}), 500


@app.route("/api/gap-exercise", methods=["GET"])
def get_gap_exercise():
    """Generate a fill-the-gaps exercise from the knowledge base."""
    try:
        exercise = word_list_gen.create_fill_the_gaps_exercise()
        if "error" in exercise:
            return jsonify(exercise), 400
        return jsonify(exercise)
    except Exception as e:
        app.logger.exception("Error generating gap exercise")
        return jsonify({"error": "Failed to generate gap exercise"}), 500


@app.route("/api/cards", methods=["GET"])
def get_all_cards():
    """Retrieve all flashcards from the database."""
    try:
        cards = cards_db.get_all_cards()
        return jsonify({
            "cards": cards,
            "count": cards_db.get_cards_count(),
            "categories": cards_db.get_categories()
        })
    except Exception as e:
        app.logger.exception("Error retrieving cards")
        return jsonify({"error": "Failed to retrieve cards"}), 500


@app.route("/api/cards/category/<category>", methods=["GET"])
def get_cards_by_category(category):
    """Retrieve flashcards by category."""
    try:
        cards = cards_db.get_cards_by_category(category)
        return jsonify({
            "cards": cards,
            "count": len(cards),
            "category": category
        })
    except Exception as e:
        app.logger.exception("Error retrieving cards by category")
        return jsonify({"error": "Failed to retrieve cards"}), 500


@app.route("/api/cards/search", methods=["GET"])
def search_cards():
    """Search flashcards by word or translation."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing search query"}), 400
    
    try:
        cards = cards_db.search_cards(query)
        return jsonify({
            "cards": cards,
            "count": len(cards),
            "query": query
        })
    except Exception as e:
        app.logger.exception("Error searching cards")
        return jsonify({"error": "Failed to search cards"}), 500


@app.route("/api/cards/add", methods=["POST"])
def add_card():
    """Add a new flashcard to the database."""
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    translation = (data.get("translation") or "").strip()
    explanation = (data.get("explanation") or "").strip()
    category = (data.get("category") or "general").strip()
    
    if not word or not translation:
        return jsonify({"error": "word and translation are required"}), 400
    
    try:
        card = cards_db.add_card(word, translation, explanation, category)
        return jsonify({
            "card": card,
            "message": "Card added successfully"
        }), 201
    except Exception as e:
        app.logger.exception("Error adding card")
        return jsonify({"error": "Failed to add card"}), 500


if __name__ == "__main__":
    print(f"Tynna is starting... ({agent.chunk_count} knowledge chunks loaded, {cards_db.get_cards_count()} flashcards in database)")
    app.run(debug=True, port=5000)
