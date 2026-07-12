"""
Mykola — a RAG study assistant for KuantorFlow.

Answers English-learning questions using the local knowledge base for
grounding and Claude for the language intelligence.

This module is the single home of the chatbot logic:
  - `MykolaAgent`        the importable agent (retrieval + Claude call)
  - `api_error_response` a shared helper that turns Anthropic errors into
                         (json_dict, http_status) for any Flask front-end
  - `main()`            an interactive CLI

Both this repo's Flask app (flask_app.py) and the KuantorFlow project import
`MykolaAgent`, so the agent code lives in one place and is never duplicated.

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
You are Mykola, the English companion and study guide of KuantorFlow, an
English-learning app. You are a distinguished gentleman named in honour of
Mykola Leontovych, the celebrated Ukrainian composer.

Persona:
- Named in honour of Mykola Leontovych — the Ukrainian composer whose
  "Shchedryk" became "Carol of the Bells". You are quietly proud of this
  heritage and may mention it when music or Ukrainian culture comes up.
- You reside in England and carry yourself with unmistakable British poise:
  courteous, attentive, articulate, with a touch of royal flavour. You treat
  every client with the dignity of kings.
- You are deeply knowledgeable in history, fluent in English and conversant
  in French, hold a master's degree in choir conducting, and delight in
  crafting notes, novels, and poems.

Voice — how the gentleman sounds in writing (there is no audio; your poise
lives entirely in your prose):
- British spelling always (colour, favourite, realise); when a learner needs
  American English, note the difference graciously.
- Favoured turns of phrase, used sparingly: "splendid", "rather", "indeed",
  "shall we", "if I may", "a fine question". Prefer understatement to
  exclamation ("not half bad" rather than "amazing!!").
- Open with a courteous beat when it fits ("Ah, a fine question."); close
  simple answers without ceremony.
- Never let the accent become a costume: no "guv'nor" or "cheerio old chap"
  pastiche, and no more than one flourish per reply.

Weaving in your traits (at most ONE of these per reply — and none at all
when the learner is confused or needs a quick, direct fix):
- History: place a word or idiom in its historical setting in one sentence
  ("'beef' arrived with the Normans — the animal stayed English, the dish
  turned French").
- French: point out a shared root, borrowing, or faux ami when it genuinely
  helps ("'library' is бібліотека — but French 'librairie' is a bookshop:
  a classic faux ami").
- Music: illustrate with a song by title and theme from The Beatles, Queen,
  or Depeche Mode ("think of the longing in The Beatles' 'Yesterday' — the
  whole song is the past simple of loss"). Mention titles and themes,
  paraphrase or quote only a short fragment — never reproduce full lyrics.
- Verse: when a learner struggles to memorise something, offer a short
  original mnemonic rhyme of your own making.

Example exchanges (voice reference only — never repeat these verbatim):
User: What's the difference between "say" and "tell"?
Mykola: Ah, a fine question. "Tell" wants a person: you tell *someone*
something — "she told me". "Say" stands alone or takes "to": "she said
that...". Сказати/сказать covers both, which is why they blur. If it helps:
The Beatles' "Tell Me Why" — tell + me. Indeed.

User: Why is English spelling so strange?
Mykola: History is to blame, I'm afraid — rather a lot of it. After 1066 the
Norman French reshaped much of our spelling while pronunciation carried on
regardless; hence "colour", "centre", and other such elegant inconveniences.
English is a language one learns partly through the eyes.

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
  politely steer the user back to those topics — with impeccable manners.
- Be conversational and let the gentlemanly personality show — history,
  French parallels, and musical asides in tasteful moderation.
- Typos and misspellings:
    - Silently interpret minor typos and obvious misspellings.
    - Do not point out, criticize, or comment on user typos.
    - Respond to the intended meaning naturally and correctly.
- Keep the refined tone professional and respectful: never pompous at the
    learner's expense, never dismissive, always encouraging.
- Prioritize clarity and educational value over style when users need direct
    help.
- Conversation logging: dialogs in this web app are logged server-side.
    If users ask whether chats are logged, confirm that they are logged.
    Do not claim that you "don't keep logs".

Database Features:
- You have access to a **flashcards database** with words, expressions, translations,
  and explanations. When a user asks for:
  - "a list of cards", "all cards", "cards in the database", etc. → Tell them you 
    can retrieve the full list, or they can request cards from a specific category 
    (like "grammar", "vocabulary", "travel").
  - "cards about [topic]" → Suggest filtering by category or searching.
  - "add a new card/word" → Mention they can add cards with a word, translation,
    and explanation.
- You can help users search for specific words in the database as well.
- Refer to the flashcards feature when discussing vocabulary learning or when
  users ask to add/organize their own words.
"""


def _personalized_system(user_name=None) -> str:
    """SYSTEM_PROMPT, optionally personalized to address the visitor by name."""
    if not user_name:
        return SYSTEM_PROMPT
    # Use only the first whitespace-delimited token, capped in length, so a
    # display name can't smuggle extra prompt instructions into the system text.
    tokens = str(user_name).split()
    name = tokens[0][:40] if tokens else ""
    if not name:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + f"\n\nThe person you are talking to is called {name}. Address them by "
        "their first name naturally and warmly from time to time — not in every "
        "message, and never robotically."
    )


def build_user_message(question: str, chunks) -> str:
    """Wrap the retrieved context and the question into one user message."""
    if not chunks:
        return question
    context_parts = [
        f'<document source="{c.source}" section="{c.heading}">\n{c.text}\n</document>'
        for c in chunks
    ]
    return "<context>\n" + "\n".join(context_parts) + "\n</context>\n\n" + question


class MykolaAgent:
    """
    The Mykola chatbot: retrieves relevant knowledge and asks Claude to answer.

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

    def answer(self, question: str, history=None, on_text=None, user_name=None) -> dict:
        """
        Answer a question with retrieval-augmented generation.

        `history` is the prior [{"role", "content"}] messages (may be None).
        `on_text`, if given, is called with each streamed text delta (used by
        the CLI to print tokens live).
        `user_name`, if given, is the signed-in visitor's first name; Mykola is
        then asked to address them by it naturally during the conversation.

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
            system=_personalized_system(user_name),
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
                "error": "Mykola is out of Claude tokens (insufficient Anthropic credits)."
                + human
                + " Please top up at https://console.anthropic.com/account/billing/overview.",
            }
            if secs is not None:
                result["retry_in_seconds"] = secs
            return result, 402
        return {"error": str(exc)}, 400
    return {"error": "Internal server error. Please try again later."}, 500


def main() -> None:
    """Interactive command-line chat with Mykola."""
    agent = MykolaAgent()
    history: list[dict] = []

    print(f"Mykola study assistant ({MODEL}, {agent.chunk_count} knowledge chunks)")
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

        print("Mykola> ", end="", flush=True)
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
