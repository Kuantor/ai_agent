"""The deck-reading tools — list_topics, list_words, get_card (#68, #73).

Plain script like the other tests here. No API key and no database: the readers
are stubbed exactly as kuantorflow's injection would supply them, which is the
whole point of the seam.
"""

import json

import agent as A


class Stub(A.MykolaAgent):
    """A MykolaAgent with the Anthropic client and knowledge base skipped."""

    def __init__(self, topic_reader=None, card_reader=None):
        self.topic_reader = topic_reader or self._default_topic_reader
        self.card_reader = card_reader or self._default_card_reader
        self._cards_db = None


DECK = {
    "Daily life and routines": [
        {"word": "chore", "pos": "noun", "explanation_en": "a routine task",
         "translation_ukr": "хатня робота", "translation_rus": "домашнее дело",
         "examples_en": ["I do my chores on Sunday."] * 3},
        {"word": "commute", "pos": "verb", "explanation_en": "to travel to work",
         "translation_ukr": "їздити на роботу"},
        # No explanation — 74 of the 503 live cards are like this.
        {"word": "errand", "pos": "noun"},
        # One word, two parts of speech: two cards, one word (#228).
        {"word": "book", "pos": "noun", "explanation_en": "a bound volume"},
        {"word": "book", "pos": "verb", "explanation_en": "to reserve"},
    ],
    "Work and careers": [{"word": "resign", "pos": "verb"}],
}


def topics():
    return [{"topic": name, "cards": len(c)} for name, c in DECK.items()]


def cards(topic, limit):
    return DECK.get(topic, [])[:limit]


ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


a = Stub(topics, cards)

print("list_topics")
out = json.loads(a._run_list_topics({}))
check("returns every topic with its count",
      out["status"] == "ok" and len(out["topics"]) == 2)
check("carries the exact names the model must choose between",
      {t["topic"] for t in out["topics"]} ==
      {"Daily life and routines", "Work and careers"})
check("an empty deck says so rather than erroring",
      json.loads(Stub(lambda: [], cards)._run_list_topics({}))["status"] == "empty")


def boom():
    raise RuntimeError("database is down")


check("a failing reader becomes an error status, never an exception",
      json.loads(Stub(boom, cards)._run_list_topics({}))["status"] == "error")

print("\nlist_words — the words, and nothing else")
out = json.loads(a._run_list_words({"topic": "Daily life and routines"}))
check("returns plain strings",
      out["status"] == "ok" and all(isinstance(w, str) for w in out["words"]))
check("no explanations", "explanation" not in json.dumps(out))
check("no translations", "translation" not in json.dumps(out))
check("no examples", "example" not in json.dumps(out))
check("no parts of speech — the word is the unit here",
      "pos" not in json.dumps(out))
check("a word filed twice appears once",
      out["words"].count("book") == 1 and out["total"] == 4)
check("carries the words themselves",
      set(out["words"]) == {"chore", "commute", "errand", "book"})

check("an unknown topic hands back the names to choose from",
      "Daily life and routines" in json.loads(
          a._run_list_words({"topic": "daily routines"}))["available_topics"])
check("a missing topic is refused",
      json.loads(a._run_list_words({}))["status"] == "error")

print("\nit is small — the point of the change")
words_only = len(json.dumps(out, ensure_ascii=False))
full = len(json.dumps({"cards": DECK["Daily life and routines"]},
                      ensure_ascii=False))
check(f"a word list is a fraction of the cards ({words_only} vs {full} chars)",
      words_only * 3 < full)

print("\nget_card — what the learner's card actually says")
out = json.loads(a._run_get_card(
    {"topic": "Daily life and routines", "word": "chore"}))
check("returns the card", out["status"] == "ok" and len(out["cards"]) == 1)
check("with its part of speech and English explanation",
      out["cards"][0]["pos"] == "noun" and
      out["cards"][0]["explanation_en"] == "a routine task")
check("never a translation", "translation" not in json.dumps(out))
check("never examples", "example" not in json.dumps(out))

out = json.loads(a._run_get_card(
    {"topic": "Daily life and routines", "word": "book"}))
check("a word with two parts of speech returns both cards",
      len(out["cards"]) == 2 and
      {c["pos"] for c in out["cards"]} == {"noun", "verb"})

out = json.loads(a._run_get_card(
    {"topic": "Daily life and routines", "word": "errand"}))
check("a card with no explanation still returns, so he can say it is empty",
      out["status"] == "ok" and "explanation_en" not in out["cards"][0])

check("matching ignores case",
      json.loads(a._run_get_card(
          {"topic": "Daily life and routines", "word": "CHORE"}
      ))["status"] == "ok")
check("a word that is not there says so",
      json.loads(a._run_get_card(
          {"topic": "Daily life and routines", "word": "zzz"}
      ))["status"] == "unknown_word")
check("an unknown topic still offers the names",
      "available_topics" in json.loads(a._run_get_card(
          {"topic": "nope", "word": "chore"})))
for missing in ({}, {"topic": "x"}, {"word": "y"}):
    check(f"both arguments are required ({sorted(missing)})",
          json.loads(a._run_get_card(missing))["status"] == "error")

print("\nno route returns a translation")
blob = "".join([
    a._run_list_topics({}),
    a._run_list_words({"topic": "Daily life and routines"}),
    a._run_get_card({"topic": "Daily life and routines", "word": "chore"}),
])
check("not from any of the three tools", "translation" not in blob)
check("the translation fields are gone from the module",
      not hasattr(A, "TRANSLATION_FIELDS"))
check("so is the tool they belonged to",
      not hasattr(A, "GET_FLASHCARDS_TOOL"))

print("\nwiring")
names = {t["name"] for t in A.TOOLS}
check("both new tools are registered", {"list_words", "get_card"} <= names)
check("the old one is not", "get_flashcards" not in names)
check("both dispatch to a handler",
      json.loads(a._run_tool("list_words",
                             {"topic": "Work and careers"}))["status"] == "ok" and
      json.loads(a._run_tool("get_card", {"topic": "Work and careers",
                                          "word": "resign"}))["status"] == "ok")
check("descriptions say when to call, not only what they do",
      "call this" in A.LIST_WORDS_TOOL["description"].lower() and
      "call this" in A.GET_CARD_TOOL["description"].lower())
check("get_card's description warns against using it as a dictionary",
      "you already know english" in A.GET_CARD_TOOL["description"].lower())

print(f"\n{ok} checks passed")
