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

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Load the knowledge base and Anthropic client once, reuse across requests.
agent = TynnaAgent()
word_list_gen = WordListGenerator(agent.kb.chunks)


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


if __name__ == "__main__":
    print(f"Tynna is starting... ({agent.chunk_count} knowledge chunks loaded)")
    app.run(debug=True, port=5000)
