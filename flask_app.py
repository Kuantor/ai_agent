"""
Flask web app for Tynna: an AI study assistant with a human touch.
Integrates RAG retrieval and Claude for English-learning help.

Run:  python flask_app.py
Needs ANTHROPIC_API_KEY in .env
"""

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from rag import KnowledgeBase

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize knowledge base and Anthropic client
kb = KnowledgeBase()
client = anthropic.Anthropic()

MODEL = "claude-opus-4-8"
MAX_TOKENS = 8192
TOP_K = 3

SYSTEM_PROMPT = """\
You are Tynna, the study assistant of KuantorFlow, an English-learning app.
You are warm, encouraging, and friendly — like a real tutor who cares about 
your progress.

Users are Ukrainian and Russian speakers learning English.

Rules:
- Answer questions about English (grammar, vocabulary, usage) clearly and 
  briefly, with 1–2 short examples.
- A <context> block with excerpts from the app's knowledge base may be 
  provided. Prefer it when relevant, and mention which document you used.
- If the context does not cover the question, say so in one short sentence 
  and answer from your general knowledge.
- When helpful, add the Ukrainian or Russian translation of key terms.
- If the question is not related to learning English or using KuantorFlow, 
  politely steer the user back to those topics.
- Be conversational and sometimes add a little personality—show you care 
  about their learning journey.
"""


def build_user_message(question: str, chunks) -> str:
    """Wrap retrieved context and question into one user message."""
    if not chunks:
        return question
    context_parts = [
        f'<document source="{c.source}" section="{c.heading}">\n{c.text}\n</document>'
        for c in chunks
    ]
    return "<context>\n" + "\n".join(context_parts) + "\n</context>\n\n" + question


@app.route("/")
def home():
    """Render the main chat page."""
    return render_template("index.html", kb_chunks_count=len(kb.chunks))


@app.route("/about")
def about():
    """Render Tynna's profile and gallery page."""
    return render_template("about.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint for chat. Accepts question, returns streamed response."""
    data = request.get_json()
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        # Retrieve relevant chunks
        chunks = kb.retrieve(question, top_k=TOP_K)
        
        # Build user message with context
        user_message = build_user_message(question, chunks)
        
        # Add to history
        history.append({"role": "user", "content": user_message})

        # Call Claude API with streaming
        response_text = ""
        thinking_text = ""
        
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=history,
        ) as stream:
            for event in stream:
                # Handle content block events
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'type'):
                            if event.delta.type == 'text_delta':
                                response_text += event.delta.text
                            elif event.delta.type == 'thinking_delta':
                                thinking_text += event.delta.thinking
            
            final_message = stream.get_final_message()

        # Prepare response with sources
        sources = []
        if chunks:
            sources = [
                {
                    "file": c.source,
                    "heading": c.heading,
                    "score": round(c.score, 2),
                }
                for c in chunks
            ]

        assistant_content = response_text
        return jsonify({
            "response": assistant_content,
            "sources": sources,
            "history": history + [{"role": "assistant", "content": assistant_content}]
        })

    except anthropic.AuthenticationError:
        return jsonify({
            "error": "Invalid or missing ANTHROPIC_API_KEY. "
                    "Please set your key in .env"
        }), 401
    except anthropic.APIConnectionError:
        return jsonify({"error": "Network error. Please try again."}), 503
    except anthropic.BadRequestError as e:
        error_msg = str(e)
        if "credit balance" in error_msg.lower():
            return jsonify({
                "error": "Anthropic account has insufficient credits. "
                        "Please top up at https://console.anthropic.com/account/billing/overview"
            }), 402
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        app.logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": "Internal server error. Please try again later."}), 500


if __name__ == "__main__":
    print(f"Tynna is starting... ({len(kb.chunks)} knowledge chunks loaded)")
    app.run(debug=True, port=5000)
