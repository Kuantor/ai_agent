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

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Load the knowledge base and Anthropic client once, reuse across requests.
agent = TynnaAgent()


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


if __name__ == "__main__":
    print(f"Tynna is starting... ({agent.chunk_count} knowledge chunks loaded)")
    app.run(debug=True, port=5000)
