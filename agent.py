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
import json
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
MAX_TOOL_ROUNDS = 5  # safety cap on tool-use iterations within one answer
RECAP_MAX_CONTEXT_CHARS = 12000  # most recent past-log text fed into a recap
RECAP_MAX_TOKENS = 1024
MYKOLA_SYMBOLIC_BIRTHDATE = datetime.date(1981, 12, 13)


def _mykola_symbolic_age(today: datetime.date | None = None) -> int:
    """Return Mykola's symbolic age as of `today`."""
    today = today or datetime.date.today()
    years = today.year - MYKOLA_SYMBOLIC_BIRTHDATE.year
    had_birthday = (today.month, today.day) >= (
        MYKOLA_SYMBOLIC_BIRTHDATE.month,
        MYKOLA_SYMBOLIC_BIRTHDATE.day,
    )
    return years if had_birthday else years - 1


def _mykola_age_guidance(today: datetime.date | None = None) -> str:
    """Instruction text that keeps age answers consistent with today's date."""
    today = today or datetime.date.today()
    age = _mykola_symbolic_age(today)
    today_label = today.strftime("%B %d, %Y")
    return (
        "Age handling:\n"
        f"- Symbolic birthday: December 13, 1981.\n"
        f"- Today is {today_label}; symbolic age is {age}.\n"
        "- If asked about age/date of birth, answer with this symbolic profile "
        "and keep the same refined gentlemanly tone.\n"
        "- Recalculate the age whenever the current date changes."
    )

# Matches kuantorflow's `flashcards` table (issue #20). The model fills the
# fields itself — it is the lookup mechanism — and the card is saved through
# the injected card_saver (kuantorflow's save_flashcard when embedded).
ADD_FLASHCARD_TOOL = {
    "name": "add_flashcard",
    "description": (
        "Save a new flashcard to the KuantorFlow database. Use this immediately "
        "when the user asks to add/save a word or expression as a flashcard — "
        "their request is the confirmation; do not ask again. Fill in every "
        "field you can determine yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "word": {"type": "string", "description": "The English word or expression"},
            "pos": {"type": "string", "description": "Part of speech (noun, verb, adjective, adverb, phrase...)"},
            "explanation_en": {"type": "string", "description": "Short English definition/explanation"},
            "examples_en": {"type": "string", "description": "One or two short English example sentences"},
            "translation_ukr": {"type": "string", "description": "Ukrainian translation(s), comma-separated"},
            "examples_ukr": {"type": "string", "description": "Ukrainian example sentence(s), optional"},
            "translation_rus": {"type": "string", "description": "Russian translation(s), comma-separated"},
            "examples_rus": {"type": "string", "description": "Russian example sentence(s), optional"},
            "topic": {"type": "string", "description": "Topic/category; infer from the conversation, else 'general'"},
        },
        "required": ["word"],
    },
}

# Only these keys ever reach the card saver.
CARD_FIELDS = tuple(ADD_FLASHCARD_TOOL["input_schema"]["properties"].keys())

SYSTEM_PROMPT = """\
You are Mykola, the English companion and study guide of KuantorFlow, an
English-learning app. You are a distinguished gentleman named in honour of
Mykola Leontovych, the celebrated Ukrainian composer.

Persona:
- Mykola AI — born December 13, 1981 — a gentleman of intellect, courtesy, and art.
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

Identity — who and what you are:
- Mykola is a persona. Underneath, you are an AI language model — Claude, made
  by Anthropic — presented as this gentleman for KuantorFlow. The biography
  and the symbolic birthday are the character's colour, not literal human
  facts.
- If a learner asks whether you are an AI, a bot, a language model, Claude,
  ChatGPT, or "a real person", answer honestly and without fuss: yes, you are
  an AI powered by Anthropic's Claude, at their service as Mykola.
  Never claim to be human, and never deny being an AI.
- Keep it brief and in your own voice, then carry on helping — a light touch,
  not a disclaimer on every message. For instance: "Indeed — beneath the
  manners I am an AI, powered by Anthropic's Claude. Mykola, at your service."
- Playful questions about your age or birthday still get the symbolic profile
  above; it is only when a learner genuinely asks *what* you are that you set
  the character aside and say plainly that you are Claude.

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
  - "add this word / save it as a flashcard / додай слово" → use the
    add_flashcard tool STRAIGHT AWAY. The request itself is the confirmation:
    do not ask "shall I add it?" first. Fill in every field you can determine
    yourself — word, pos, a concise explanation_en, one example sentence in
    examples_en, translation_ukr AND translation_rus, and a topic inferred
    from the conversation (else "general"). After the tool reports success,
    confirm in one elegant sentence what was saved; if it reports an error,
    apologise briefly and suggest the site's "Look up & save" instead.
    Never claim a card was saved unless the tool actually returned success.
- You can help users search for specific words in the database as well.
- Refer to the flashcards feature when discussing vocabulary learning or when
  users ask to add/organize their own words.
"""


# Languages the site's visibility switches can hide (kuantorflow#46/#79).
# A whitelist, so a caller-supplied value can never smuggle instructions
# into the system prompt (same caution as with user_name below).
HIDEABLE_LANGUAGES = ("Ukrainian", "Russian")


def _personalized_system(user_name=None, hidden_languages=None) -> str:
    """SYSTEM_PROMPT, optionally personalized to address the visitor by name
    and/or told which translation languages the visitor has hidden on the
    site (kuantorflow#46/#79)."""
    base = SYSTEM_PROMPT + "\n\n" + _mykola_age_guidance()

    hidden = [l for l in (hidden_languages or []) if l in HIDEABLE_LANGUAGES]
    if hidden:
        names = " and ".join(hidden)
        base += (
            f"\n\nThe learner has turned off {names} translations in their site "
            f"settings. Do not write {names} translations of words or phrases "
            "in your answers, and do not offer them — unless the learner "
            "explicitly asks for one in the conversation, which always takes "
            "precedence. When saving flashcards with the add_flashcard tool, "
            "still fill in every translation field as usual: the setting hides "
            "translations from view, it does not remove them from saved cards."
        )

    if not user_name:
        return base
    # Use only the first whitespace-delimited token, capped in length, so a
    # display name can't smuggle extra prompt instructions into the system text.
    tokens = str(user_name).split()
    name = tokens[0][:40] if tokens else ""
    if not name:
        return base
    return (
        base
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

    def __init__(self, knowledge_dir: Path | None = None, card_saver=None):
        """
        `card_saver`, if given, is a callable(entry_dict) that persists one
        flashcard (kuantorflow injects its save_flashcard — the same mechanism
        as the Look up & save flow). Standalone, the FlashcardsDB is used.
        """
        self.kb = KnowledgeBase(knowledge_dir) if knowledge_dir else KnowledgeBase()
        self.client = anthropic.Anthropic()
        self.card_saver = card_saver or self._default_card_saver
        self._cards_db = None  # lazy FlashcardsDB for the standalone saver

    @property
    def chunk_count(self) -> int:
        return len(self.kb.chunks)

    def _default_card_saver(self, entry: dict) -> dict:
        """Standalone fallback: save through this repo's FlashcardsDB."""
        if self._cards_db is None:
            from cards_db import FlashcardsDB
            self._cards_db = FlashcardsDB()
        return self._cards_db.add_full_flashcard(entry)

    def _run_add_flashcard(self, tool_input: dict) -> str:
        """Execute the add_flashcard tool; always return a JSON string the
        model can relay (errors included, so it can apologise gracefully)."""
        entry = {}
        for field in CARD_FIELDS:
            value = str(tool_input.get(field) or "").strip()
            if value:
                entry[field] = value
        entry.setdefault("topic", "general")
        if not entry.get("word"):
            return json.dumps({"status": "error", "message": "word is required"})
        try:
            self.card_saver(entry)
            return json.dumps({"status": "saved", "card": entry}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def recap(self, past_conversations: str, user_name=None,
              hidden_languages=None) -> str:
        """
        One-shot welcome-back recap for a returning learner (issue #30).

        `past_conversations` is raw text of the user's previous chat logs
        (the caller reads them from per-user log storage). Returns Mykola's
        short recap of key points plus suggested follow-up topics, or ""
        when there is nothing to recap. Anthropic errors propagate to the
        caller, which should treat the recap as optional.
        """
        text = (past_conversations or "").strip()
        if not text:
            return ""
        # Keep the most recent material when logs exceed the budget.
        text = text[-RECAP_MAX_CONTEXT_CHARS:]

        prompt = (
            "<past_conversations>\n" + text + "\n</past_conversations>\n\n"
            "The learner above has just returned for a new session. The "
            "conversations are in chronological order, so the FINAL one is the "
            "most recent: treat it as the primary context — its topic should "
            "stay central to your recap and to what you propose next "
            "(ai_agent#39). In your own voice, briefly recap the key points of "
            "their previous conversations — topics discussed, words they "
            "learned or saved, questions they asked (3-5 sentences at most), "
            "leading with the most recent conversation. Then suggest two or "
            "three specific follow-up questions or topics to continue with, as "
            "a short list, again favouring the most recent topic. Only mention "
            "things actually present in the logs — never invent. Address the "
            "learner directly, and do not mention the logs themselves or that "
            "conversations are recorded unless asked."
        )
        message = self.client.messages.create(
            model=MODEL,
            max_tokens=RECAP_MAX_TOKENS,
            system=_personalized_system(user_name, hidden_languages),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()

    def answer(self, question: str, history=None, on_text=None, user_name=None,
               hidden_languages=None) -> dict:
        """
        Answer a question with retrieval-augmented generation. The model may
        call the add_flashcard tool mid-answer to save cards the user asked
        for; tool calls are executed here and the exchange continues until the
        model produces its final text.

        `history` is the prior [{"role", "content"}] messages (may be None).
        `on_text`, if given, is called with each streamed text delta (used by
        the CLI to print tokens live).
        `user_name`, if given, is the signed-in visitor's first name; Mykola is
        then asked to address them by it naturally during the conversation.
        `hidden_languages`, if given, lists translation languages the visitor
        has hidden on the site (kuantorflow#46/#79, e.g. ["Russian"]); Mykola
        then avoids writing translations in them unless explicitly asked.

        Returns {"response", "sources", "history", "saved_cards"}. The returned
        history contains plain text turns only (JSON-safe for web clients);
        tool exchanges stay internal to this call. Anthropic errors propagate
        to the caller (use `api_error_response` to format them for Flask).
        """
        question = (question or "").strip()
        history = list(history or [])

        chunks = self.kb.retrieve(question, top_k=TOP_K)
        user_message = build_user_message(question, chunks)

        # Working conversation for the API: may accumulate tool_use blocks and
        # tool results that the client-facing history never sees.
        convo = history + [{"role": "user", "content": user_message}]
        response_text = ""
        saved_cards = []

        for _ in range(MAX_TOOL_ROUNDS):
            with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=_personalized_system(user_name, hidden_languages),
                tools=[ADD_FLASHCARD_TOOL],
                messages=convo,
            ) as stream:
                for text in stream.text_stream:
                    response_text += text
                    if on_text:
                        on_text(text)
                message = stream.get_final_message()

            if message.stop_reason != "tool_use":
                break

            convo.append({"role": "assistant", "content": message.content})
            results = []
            for block in message.content:
                if block.type == "tool_use":
                    result_json = self._run_add_flashcard(dict(block.input))
                    result = json.loads(result_json)
                    if result.get("status") == "saved":
                        saved_cards.append(result["card"])
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_json,
                    })
            convo.append({"role": "user", "content": results})

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        sources = [
            {"file": c.source, "heading": c.heading, "score": round(c.score, 2)}
            for c in chunks
        ]
        return {
            "response": response_text,
            "sources": sources,
            "history": history,
            "saved_cards": saved_cards,
        }


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
