"""One log line per tool call (#75).

The point of the line is to make the read tools' *choices* observable — which
tool the model reached for, and what it got — so #73's bet on prescriptive tool
descriptions can be checked against real conversations instead of assumed.
"""

import json
import logging

import agent as A

records = []
handler = logging.Handler()
handler.emit = records.append
log = logging.getLogger("mykola.tools")
log.addHandler(handler)
log.setLevel(logging.INFO)

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


def lines():
    return [r.getMessage() for r in records]


DECK = {"Daily life and routines": [
    {"word": "chore", "pos": "noun", "explanation_en": "a routine task"},
    {"word": "commute", "pos": "verb", "explanation_en": "to travel to work"},
]}


class Stub(A.MykolaAgent):
    def __init__(self):
        self.topic_reader = lambda: [{"topic": t, "cards": len(c)}
                                     for t, c in DECK.items()]
        self.card_reader = lambda t, limit: DECK.get(t, [])[:limit]
        self._cards_db = None


a = Stub()

print("every read is logged, through the one funnel")
a._run_tool("list_topics", {})
check("list_topics logs, with a count",
      "tool=list_topics" in lines()[-1] and "topics=1" in lines()[-1])

a._run_tool("list_words", {"topic": "Daily life and routines"})
check("list_words logs the topic and how many words came back",
      "tool=list_words" in lines()[-1] and
      "'Daily life and routines'" in lines()[-1] and "words=2" in lines()[-1])

a._run_tool("get_card", {"topic": "Daily life and routines", "word": "chore"})
check("get_card logs the word too",
      "tool=get_card" in lines()[-1] and "word='chore'" in lines()[-1] and
      "cards=1" in lines()[-1])

print("\nthe failure paths are the interesting ones")
a._run_tool("list_words", {"topic": "daily routines"})
check("a mis-guessed topic name is visible as unknown_topic",
      "status='unknown_topic'" in lines()[-1])
check("...which is how you find out whether model-side matching works",
      "'daily routines'" in lines()[-1])

a._run_tool("get_card", {"topic": "Daily life and routines", "word": "zzz"})
check("a word that is not there is visible too",
      "status='unknown_word'" in lines()[-1])

a._run_tool("nonsense", {})
check("an unknown tool is not logged as a read",
      "tool=nonsense" not in "".join(lines()))

print("\nno identity, deliberately (#163)")
blob = "".join(lines())
for leak in ("user", "email", "name=", "session", "id="):
    check(f"nothing resembling an identity in the line ({leak!r})",
          leak not in blob)

print("\nit cannot break a conversation")
before = len(records)
A._log_tool_use("x", {"topic": object()}, "not json at all")
A._log_tool_use("x", None, None)
check("malformed input never raises", True)
check("...and a bad result still produces a line rather than nothing",
      len(records) > before)

print("\nthe call still returns its result unchanged")
out = json.loads(a._run_tool("get_card", {"topic": "Daily life and routines",
                                          "word": "commute"}))
check("logging does not swallow or alter the tool result",
      out["status"] == "ok" and out["cards"][0]["word"] == "commute")

print(f"\n{ok} checks passed")
