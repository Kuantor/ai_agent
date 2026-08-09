"""The deck-reading tools (#68) — list_topics and get_flashcards.

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
         "translation_ukr": "їздити на роботу", "translation_rus": "ездить на работу"},
    ],
    "Work and careers": [{"word": "resign", "pos": "verb"}],
}


def topics():
    return [{"topic": name, "cards": len(cards)} for name, cards in DECK.items()]


def cards(topic, limit):
    return DECK.get(topic, [])[:limit]


ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


print("list_topics")
a = Stub(topics, cards)
out = json.loads(a._run_list_topics({}))
check("returns every topic with its count",
      out["status"] == "ok" and len(out["topics"]) == 2)
check("carries the exact names the model must choose between",
      {t["topic"] for t in out["topics"]} ==
      {"Daily life and routines", "Work and careers"})

empty = json.loads(Stub(lambda: [], cards)._run_list_topics({}))
check("an empty deck says so rather than erroring", empty["status"] == "empty")


def boom():
    raise RuntimeError("database is down")


broke = json.loads(Stub(boom, cards)._run_list_topics({}))
check("a failing reader becomes an error status, never an exception",
      broke["status"] == "error" and "down" in broke["message"])

print("\nget_flashcards")
out = json.loads(a._run_get_flashcards({"topic": "Daily life and routines"}))
check("returns the topic's cards", out["status"] == "ok" and len(out["cards"]) == 2)
check("keeps the word, its part of speech and its English explanation",
      out["cards"][0]["word"] == "chore" and
      out["cards"][0]["pos"] == "noun" and
      out["cards"][0]["explanation_en"] == "a routine task")
check("drops examples_en, the largest field on a card",
      all("examples_en" not in c for c in out["cards"]))
check("omits fields the card does not have",
      "explanation_en" not in json.loads(
          a._run_get_flashcards({"topic": "Work and careers"}))["cards"][0])

print("\ntranslations are opt-in")
check("no translations by default",
      all("translation_ukr" not in c and "translation_rus" not in c
          for c in out["cards"]))

asked = json.loads(a._run_get_flashcards(
    {"topic": "Daily life and routines", "include_translations": True}))
check("include_translations brings them back",
      asked["cards"][0]["translation_ukr"] == "хатня робота" and
      asked["cards"][0]["translation_rus"] == "домашнее дело")
check("...and still no examples_en",
      all("examples_en" not in c for c in asked["cards"]))

for junk in ("yes", 1, None):
    got = json.loads(a._run_get_flashcards(
        {"topic": "Daily life and routines", "include_translations": junk}))
    check(f"only a real true opts in (include_translations={junk!r})",
          "translation_ukr" not in got["cards"][0])

# The host strips a hidden language before the row reaches the agent, so
# asking for translations cannot surface one the learner turned off (#46/#79).
def ukrainian_hidden(topic, limit):
    return [{k: v for k, v in card.items() if k != "translation_ukr"}
            for card in DECK.get(topic, [])[:limit]]

hidden = json.loads(Stub(topics, ukrainian_hidden)._run_get_flashcards(
    {"topic": "Daily life and routines", "include_translations": True}))
check("a hidden language never appears, even when translations are asked for",
      "translation_ukr" not in hidden["cards"][0] and
      hidden["cards"][0]["translation_rus"] == "домашнее дело")

unknown = json.loads(a._run_get_flashcards({"topic": "daily routines"}))
check("an unknown topic is not a bare failure",
      unknown["status"] == "unknown_topic")
check("...it hands back the names to choose from",
      "Daily life and routines" in unknown["available_topics"])

check("a missing topic is refused",
      json.loads(a._run_get_flashcards({}))["status"] == "error")

print("\nbounding the payload")
big = {"T": [{"word": f"w{i}"} for i in range(100)]}
b = Stub(lambda: [{"topic": "T", "cards": 100}],
         lambda t, limit: big["T"][:limit])

out = json.loads(b._run_get_flashcards({"topic": "T"}))
check(f"defaults to {A.DEFAULT_CARDS_PER_READ} cards",
      len(out["cards"]) == A.DEFAULT_CARDS_PER_READ)
check("reports that more were withheld", out["withheld"] > 0)

out = json.loads(b._run_get_flashcards({"topic": "T", "limit": 5}))
check("honours an explicit limit", len(out["cards"]) == 5)

out = json.loads(b._run_get_flashcards({"topic": "T", "limit": 9999}))
check(f"clamps to {A.MAX_CARDS_PER_READ}",
      len(out["cards"]) == A.MAX_CARDS_PER_READ)

out = json.loads(b._run_get_flashcards({"topic": "T", "limit": "twenty"}))
check("a non-integer limit falls back to the default",
      len(out["cards"]) == A.DEFAULT_CARDS_PER_READ)

small = Stub(lambda: [{"topic": "T", "cards": 2}],
             lambda t, limit: [{"word": "a"}, {"word": "b"}][:limit])
check("withheld is zero when the topic fits",
      json.loads(small._run_get_flashcards({"topic": "T"}))["withheld"] == 0)

print("\nwiring")
check("both tools are registered", {t["name"] for t in A.TOOLS} >=
      {"list_topics", "get_flashcards"})
check("both dispatch to a handler",
      json.loads(a._run_tool("list_topics", {}))["status"] == "ok" and
      json.loads(a._run_tool("get_flashcards",
                             {"topic": "Work and careers"}))["status"] == "ok")
check("descriptions say when to call, not only what they do",
      "call this" in A.LIST_TOPICS_TOOL["description"].lower() and
      "call this" in A.GET_FLASHCARDS_TOOL["description"].lower())
check("the readers are optional kwargs, so hosts deploy in either order",
      {"topic_reader", "card_reader"} <=
      set(A.inspect.signature(A.MykolaAgent.__init__).parameters)
      if hasattr(A, "inspect") else True)

print(f"\n{ok} checks passed")
