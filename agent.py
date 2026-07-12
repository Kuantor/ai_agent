"""
Tynna — a RAG study assistant for KuantorFlow.

Answers English-learning questions using the local knowledge base for
grounding and Claude for the language intelligence.

This module is the single home of the chatbot logic:
  - `TynnaAgent`        the importable agent (retrieval + Claude call)
  - `api_error_response` a shared helper that turns Anthropic errors into
                         (json_dict, http_status) for any Flask front-end
  - `main()`            an interactive CLI

Both this repo's Flask app (flask_app.py) and the KuantorFlow project import
`TynnaAgent`, so the agent code lives in one place and is never duplicated.

Run the CLI:  python agent.py   (needs ANTHROPIC_API_KEY in .env)
"""

import datetime
import email.utils as eut
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag import KnowledgeBase

load_dotenv(Path(__file__).with_name(".env"))

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
  briefly, with 1-2 short examples.
- A <context> block with excerpts from the app's knowledge base may be
  provided. Prefer it when relevant, and mention which document you used.
- If the context does not cover the question, say so in one short sentence
  and answer from your general knowledge.
- When helpful, add the Ukrainian or Russian translation of key terms.
- If the question is not related to learning English or using KuantorFlow,
  politely steer the user back to those topics.
- Be conversational and sometimes add a little personality — show you care
  about their learning journey.
"""


def build_user_message(question: str, chunks) -> str:
    """Wrap the retrieved context and the question into one user message."""
    if not chunks:
        return question
    context_parts = [
        f'<document source="{c.source}" section="{c.heading}">\n{c.text}\n</document>'
        for c in chunks
    ]
    return "<context>\n" + "\n".join(context_parts) + "\n</context>\n\n" + question


class TynnaAgent:
    """
    The Tynna chatbot: retrieves relevant knowledge and asks Claude to answer.

    A single instance loads the knowledge base and the Anthropic client once
    and can be reused across requests. Import and reuse this class rather than
    copying the logic.
    """

    def __init__(self, knowledge_dir: Path | None = None):
        self.kb = KnowledgeBase(knowledge_dir) if knowledge_dir else KnowledgeBase()
        self.client = anthropic.Anthropic()

    @property
    def chunk_count(self) -> int:
        return len(self.kb.chunks)

    def answer(self, question: str, history=None, on_text=None) -> dict:
        """
        Answer a question with retrieval-augmented generation.

        `history` is the prior [{"role", "content"}] messages (may be None).
        `on_text`, if given, is called with each streamed text delta (used by
        the CLI to print tokens live).

        Returns {"response", "sources", "history"} where `history` includes the
        new user and assistant turns. Anthropic errors propagate to the caller
        (use `api_error_response` to format them for a Flask response).
        """
        question = (question or "").strip()
        history = list(history or [])

        chunks = self.kb.retrieve(question, top_k=TOP_K)
        history.append({"role": "user", "content": build_user_message(question, chunks)})

        response_text = ""
        with self.client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                response_text += text
                if on_text:
                    on_text(text)
            stream.get_final_message()

        history.append({"role": "assistant", "content": response_text})
        sources = [
            {"file": c.source, "heading": c.heading, "score": round(c.score, 2)}
            for c in chunks
        ]
        return {"response": response_text, "sources": sources, "history": history}


def _extract_retry_seconds(exc) -> int | None:
    """Best-effort read of a Retry-After / rate-limit-reset header from an error."""
    for attr in ("response", "raw_response", "resp"):
        resp = getattr(exc, attr, None)
        headers = getattr(resp, "headers", None) if resp else None
        if not headers:
            continue
        for key in ("Retry-After", "retry-after"):
            if key in headers:
                try:
                    return int(headers[key])
                except (TypeError, ValueError):
                    try:
                        dt = eut.parsedate_to_datetime(headers[key])
                        secs = int((dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
                        return max(0, secs)
                    except (TypeError, ValueError):
                        pass
        for key in ("x-rate-limit-reset", "x-ratelimit-reset", "x-reset"):
            if key in headers:
                try:
                    reset = int(headers[key])
                    if reset > 1e12:
                        reset = reset / 1000
                    return max(0, int(reset - time.time()))
                except (TypeError, ValueError):
                    pass
    return None


def api_error_response(exc):
    """
    Turn an Anthropic exception into (json_dict, http_status) for a Flask
    front-end. Shared by every UI so error handling isn't duplicated.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        return {"error": "Invalid or missing ANTHROPIC_API_KEY. Set your key in the ai_agent .env."}, 401
    if isinstance(exc, anthropic.APIConnectionError):
        return {"error": "Network error reaching Claude. Please try again."}, 503
    if isinstance(exc, anthropic.BadRequestError):
        text = str(exc).lower()
        if "credit balance" in text or "insufficient credits" in text or ("credit" in text and "balance" in text):
            secs = _extract_retry_seconds(exc)
            human = ""
            if secs and secs > 0:
                m, s = divmod(secs, 60)
                human = f" Try again in {m}m {s}s."
            result = {
                "error": "Tynna is out of Claude tokens (insufficient Anthropic credits)."
                + human
                + " Please top up at https://console.anthropic.com/account/billing/overview.",
            }
            if secs is not None:
                result["retry_in_seconds"] = secs
            return result, 402
        return {"error": str(exc)}, 400
    return {"error": "Internal server error. Please try again later."}, 500


def main() -> None:
    """Interactive command-line chat with Tynna."""
    agent = TynnaAgent()
    history: list[dict] = []

    print(f"Tynna study assistant ({MODEL}, {agent.chunk_count} knowledge chunks)")
    print("Ask about English grammar, vocabulary, or the app. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        print("Tynna> ", end="", flush=True)
        try:
            result = agent.answer(
                question, history, on_text=lambda t: print(t, end="", flush=True)
            )
        except anthropic.AuthenticationError:
            sys.exit("\nError: invalid or missing ANTHROPIC_API_KEY — set it in .env.")
        except (anthropic.APIConnectionError, anthropic.BadRequestError) as e:
            body, _ = api_error_response(e)
            print(f"\n[{body['error']}]")
            continue
        history = result["history"]
        print("\n")


if __name__ == "__main__":
    main()
