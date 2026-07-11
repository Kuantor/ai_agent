"""
A primitive RAG agent: answers English-learning questions using the local
knowledge base for grounding and Claude for the language intelligence.

Run:  python agent.py          (interactive chat; Ctrl+C or 'exit' to quit)
Needs ANTHROPIC_API_KEY in .env (see .env.example).
"""

import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag import KnowledgeBase

load_dotenv(Path(__file__).with_name(".env"))

MODEL = "claude-opus-4-8"
MAX_TOKENS = 8192
TOP_K = 3

SYSTEM_PROMPT = """\
You are the study assistant of KuantorFlow, an English-learning app whose
users are Ukrainian and Russian speakers learning English.

Rules:
- Answer questions about English (grammar, vocabulary, usage) clearly and
  briefly, with one or two short examples.
- A <context> block with excerpts from the app's knowledge base may be
  provided. Prefer it when relevant, and mention which document you used.
- If the context does not cover the question, say so in one short sentence
  and answer from your general knowledge.
- When helpful, add the Ukrainian or Russian translation of key terms.
- If the question is not related to learning English or using KuantorFlow,
  politely steer the user back to those topics.
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


def main() -> None:
    kb = KnowledgeBase()
    client = anthropic.Anthropic()
    history: list[dict] = []

    print(f"KuantorFlow study assistant ({MODEL}, {len(kb.chunks)} knowledge chunks)")
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

        chunks = kb.retrieve(question, top_k=TOP_K)
        if chunks:
            sources = ", ".join(f"{c.source}#{c.heading} ({c.score:.2f})" for c in chunks)
            print(f"[retrieved: {sources}]")
        else:
            print("[retrieved: nothing relevant — answering from general knowledge]")

        history.append({"role": "user", "content": build_user_message(question, chunks)})

        print("agent> ", end="", flush=True)
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=history,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                response = stream.get_final_message()
        except anthropic.AuthenticationError:
            sys.exit(
                "\nError: invalid or missing ANTHROPIC_API_KEY — "
                "copy .env.example to .env and set your key."
            )
        except anthropic.APIConnectionError:
            print("\n[network error — try again]")
            history.pop()
            continue

        print("\n")
        # Keep the full content (incl. thinking blocks) for correct multi-turn replay.
        history.append({"role": "assistant", "content": response.content})


if __name__ == "__main__":
    main()
