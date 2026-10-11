"""Mykola never answers with an empty reply when the tool rounds run out (#90).

`stream_answer()` allows MAX_TOOL_ROUNDS model calls per answer. A model that
still wanted a tool on the last one used to get nothing more: the loop ended,
that round's tools never ran, and the model never wrote an answer -- so the
learner got an empty bubble, and the empty turn went into the history. "Quiz
me on all my topics" reaches it honestly: `list_topics`, then one `list_words`
per topic, called one or two at a time.

Pinned here, offline (no key, no network):

* the **last permitted round is asked to answer** (`tool_choice` none), so it
  writes text from the tool results it already has, and is **told** the tools
  are done -- without that, measured live, it answered "Let me gather the
  words." and stopped;
* the earlier rounds are untouched -- no `tool_choice`, so a one- or two-round
  answer is exactly what it was;
* if the reply is **still** empty for any reason, Mykola says a fixed sentence
  instead of nothing, and that is what the history keeps;
* both are logged to mykola.log, so how often the cap is reached can be read
  rather than guessed (`TOOL-ROUNDS-EXHAUSTED`, `EMPTY-REPLY`).

Run: python test_tool_rounds.py
"""

import logging

import agent as A
from agent import MykolaAgent
from rag import Chunk

records = []
handler = logging.Handler()
handler.emit = records.append
for name in ("mykola.tools", "mykola.usage"):
    logging.getLogger(name).addHandler(handler)
    logging.getLogger(name).setLevel(logging.INFO)

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


def lines():
    return [r.getMessage() for r in records]


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, n):
        self.id = f"toolu_{n}"
        self.name = "list_topics"
        self.input = {}


class _Message:
    def __init__(self, blocks, stop_reason):
        self.content = blocks
        self.stop_reason = stop_reason


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return (b.text for b in self._message.content if b.type == "text")

    def get_final_message(self):
        return self._message


class _Messages:
    """A model that calls a tool on every round it is allowed to.

    `obeys` is what the real API does with `tool_choice` none: answer in text.
    A model that does not obey stands for any other way of ending up with no
    text, which the guard behind the fix has to catch.
    """

    def __init__(self, tool_rounds, obeys=True):
        self.calls = []
        self.tool_rounds = tool_rounds
        self.obeys = obeys

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        forced = kwargs.get("tool_choice") == {"type": "none"}
        if n <= self.tool_rounds and not (forced and self.obeys):
            return _Stream(_Message([_ToolUse(n)], "tool_use"))
        return _Stream(_Message([_Text(f"Here is what I found ({n}).")],
                                "end_turn"))


class _Client:
    def __init__(self, messages):
        self.messages = messages


class _KB:
    def retrieve(self, question, top_k=None):
        return [Chunk("user-guide.md", "Topics", "Guide text.", 0.5)]


def _agent(tool_rounds, obeys=True):
    agent = MykolaAgent.__new__(MykolaAgent)
    agent.kb = _KB()
    agent.client = _Client(_Messages(tool_rounds, obeys))
    agent.card_saver = None
    agent.name_saver = None
    agent._cards_db = None
    agent.topic_reader = lambda: [{"topic": "Work", "cards": 3}]
    agent.card_reader = lambda topic, limit: []
    return agent


def main():
    rounds = A.MAX_TOOL_ROUNDS

    print("a model that wants a tool on every round")
    agent = _agent(tool_rounds=99)
    del records[:]
    result = agent.answer("Quiz me on all my topics")
    calls = agent.client.messages.calls
    check(f"it is called exactly MAX_TOOL_ROUNDS ({rounds}) times",
          len(calls) == rounds)
    check("the last call asks for an answer, not a tool",
          calls[-1].get("tool_choice") == {"type": "none"})
    check("no earlier call carries tool_choice",
          all("tool_choice" not in c for c in calls[:-1]))
    check("the tools of every earlier round ran, and their results are sent",
          sum(1 for m in calls[-1]["messages"] if m["role"] == "user"
              and isinstance(m["content"], list)
              and any(isinstance(b, dict) and b.get("type") == "tool_result"
                      for b in m["content"])) == rounds - 1)
    last_user = [m for m in calls[-1]["messages"] if m["role"] == "user"][-1]
    check("the last round is told the tools are done, beside the results",
          isinstance(last_user["content"], list)
          and last_user["content"][-1].get("type") == "text"
          and last_user["content"][-1].get("text") == A.FINAL_ROUND_NOTE
          and any(isinstance(b, dict) and b.get("type") == "tool_result"
                  for b in last_user["content"]))
    check("no earlier round is told so",
          not any(isinstance(m["content"], list) and any(
              isinstance(b, dict) and b.get("text") == A.FINAL_ROUND_NOTE
              for b in m["content"])
              for c in calls[:-1] for m in c["messages"]))
    check("the reply is the model's own answer, not empty",
          result["response"] == f"Here is what I found ({rounds}).")
    check("the history keeps that answer",
          result["history"][-1] == {"role": "assistant",
                                    "content": result["response"]})
    check("the cap being reached is logged",
          any(l.startswith("TOOL-ROUNDS-EXHAUSTED") and f"rounds={rounds}" in l
              for l in lines()))

    print("and if the reply is still empty, the guard behind it")
    agent = _agent(tool_rounds=99, obeys=False)
    del records[:]
    streamed = []
    result = agent.answer("Quiz me on all my topics", on_text=streamed.append)
    check("Mykola says the fixed sentence instead of nothing",
          result["response"] == A.EMPTY_REPLY)
    check("it reaches the learner through the stream, like any answer",
          "".join(streamed) == A.EMPTY_REPLY)
    check("the history keeps the sentence, never an empty turn",
          result["history"][-1]["content"] == A.EMPTY_REPLY)
    check("the empty reply is logged with the stop reason",
          any(l.startswith("EMPTY-REPLY") and "stop=tool_use" in l
              for l in lines()))

    print("normal answers are unchanged")
    for tool_rounds, label in ((0, "one round"), (1, "two rounds")):
        agent = _agent(tool_rounds=tool_rounds)
        del records[:]
        result = agent.answer("What is a chore?")
        calls = agent.client.messages.calls
        check(f"{label}: {tool_rounds + 1} call(s), none with tool_choice",
              len(calls) == tool_rounds + 1
              and all("tool_choice" not in c for c in calls))
        check(f"{label}: the model's answer, and nothing logged about it",
              result["response"] == f"Here is what I found ({tool_rounds + 1}).")
        check(f"{label}: no exhausted or empty line",
              not any(l.startswith(("TOOL-ROUNDS-EXHAUSTED", "EMPTY-REPLY"))
                      for l in lines()))

    print("the guard's sentence is in Mykola's voice and never empty")
    check("EMPTY_REPLY is a real sentence",
          isinstance(A.EMPTY_REPLY, str) and len(A.EMPTY_REPLY.split()) > 5)

    print(f"\n{ok} checks passed")


if __name__ == "__main__":
    main()
