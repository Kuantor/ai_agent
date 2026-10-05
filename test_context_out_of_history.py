"""The guide excerpts travel with their own turn only (#88).

Each question is answered with the knowledge-base sections retrieved for it,
wrapped in a `<context>` block in front of the question. Until #88 that same
wrapped message went into the returned history, the widget stored it and sent
it back with every later message: after ten questions, up to thirty sections
about questions already answered, all still "the context" the prompt tells the
model to prefer.

Pinned here, offline (no key, no network):

* the returned history holds the **bare question**;
* the request carries `<context>` **once**, on the turn being answered;
* a history stored **before** the change has its old excerpts stripped;
* the conversation cache (#71) still works: the request marks the end of the
  stored history, which is byte-identical from one turn to the next, so the
  whole earlier conversation is read from cache.

Run: python test_context_out_of_history.py
"""

import agent as A
from agent import MykolaAgent
from rag import Chunk

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, text):
        self.content = [_TextBlock(text)]
        self.stop_reason = "end_turn"


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return (b.text for b in self._message.content)

    def get_final_message(self):
        return self._message


class _Messages:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(_Message(f"Answer {len(self.calls)}."))


class _Client:
    def __init__(self):
        self.messages = _Messages()


class _KB:
    """Returns one section for every question, named after the question."""

    def retrieve(self, question, top_k=None):
        return [Chunk("user-guide.md", f"About {question}",
                      f"Guide text about {question}.", 0.5)]


def _agent():
    agent = MykolaAgent.__new__(MykolaAgent)
    agent.kb = _KB()
    agent.client = _Client()
    agent.card_saver = None
    agent.name_saver = None
    agent._cards_db = None
    return agent


def _text(message):
    """A message's text, whether it is a string or blocks."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def _cached(message):
    content = message["content"]
    return (isinstance(content, list) and isinstance(content[-1], dict)
            and content[-1].get("cache_control") == {"type": "ephemeral"})


def main():
    agent = _agent()

    print("the returned history holds the bare question")
    first = agent.answer("How do I add a word?")
    check("the question is stored as typed",
          first["history"][0] == {"role": "user",
                                  "content": "How do I add a word?"})
    check("no <context> in the stored history",
          not any("<context>" in _text(m) for m in first["history"]))
    check("the sources are still reported for the answer",
          first["sources"] and first["sources"][0]["file"] == "user-guide.md")

    print("\nthe request carries <context> once, on the turn being answered")
    second = agent.answer("And on the phone?", first["history"])
    sent = agent.client.messages.calls[-1]["messages"]
    with_context = [m for m in sent if "<context>" in _text(m)]
    check("exactly one message carries excerpts", len(with_context) == 1)
    check("...and it is the last one, the question being answered",
          with_context[0] is sent[-1])
    check("its excerpts are the ones retrieved for this question",
          "Guide text about And on the phone?" in _text(sent[-1]))
    check("the first question's excerpts are not re-sent",
          "Guide text about How do I add a word?" not in
          "".join(_text(m) for m in sent))
    check("the history after two turns is the two questions and two answers",
          [m["content"] for m in second["history"]] ==
          ["How do I add a word?", "Answer 1.", "And on the phone?", "Answer 2."])

    print("\na history stored before #88 is stripped of its old excerpts")
    old_history = [
        {"role": "user", "content": A.build_user_message(
            "What is Review?", [Chunk("user-guide.md", "Review",
                                      "Old excerpt text.", 0.4)])},
        {"role": "assistant", "content": "Review is..."},
    ]
    check("the fixture really has the old shape",
          old_history[0]["content"].startswith("<context>\n"))
    agent.answer("Thanks", old_history)
    sent = agent.client.messages.calls[-1]["messages"]
    check("the old excerpt is not sent", "Old excerpt text." not in
          "".join(_text(m) for m in sent))
    check("the old question survives, as typed",
          _text(sent[0]) == "What is Review?")
    check("the caller's list is not modified",
          old_history[0]["content"].startswith("<context>\n"))
    check("a question merely mentioning <context> later is left alone",
          A._bare_question("what does <context> mean?") ==
          "what does <context> mean?")

    print("\nthe conversation cache (#71) still reads the earlier turns")
    agent = _agent()
    one = agent.answer("First?")
    two = agent.answer("Second?", one["history"])
    agent.answer("Third?", two["history"])
    calls = agent.client.messages.calls
    second_sent, third_sent = calls[1]["messages"], calls[2]["messages"]
    check("the end of the stored history is marked",
          _cached(second_sent[len(one["history"]) - 1]))
    check("the turn being answered is marked too (tool rounds read it)",
          _cached(second_sent[-1]))
    check("three breakpoints at most: system, history end, convo end",
          sum(_cached(m) for m in third_sent) == 2)
    # What makes the cache hit: the third request begins with exactly what the
    # second request marked, so the entry written then is the prefix read now.
    marked_prefix = second_sent[:len(one["history"])]
    check("turn 3 repeats turn 2's cached prefix byte for byte",
          [_text(m) for m in third_sent[:len(marked_prefix)]] ==
          [_text(m) for m in marked_prefix])
    check("...and its history-end marker covers turn 2's exchange",
          _cached(third_sent[len(two["history"]) - 1]))
    check("a first question has no history to mark",
          sum(_cached(m) for m in calls[0]["messages"]) == 1)

    print(f"\n{ok} checks passed")


if __name__ == "__main__":
    main()
