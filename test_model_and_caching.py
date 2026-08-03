"""
Offline checks for the Opus 5 move (#63) and the cached system prompt (#64).
Run:  python test_model_and_caching.py

No network: the Anthropic client is replaced by a stand-in that records the
keyword arguments each call site sends and hands back a canned message. What
is worth pinning here is not the wording but the *shape* of the request —
which thinking mode each call site asks for, and what sits on either side of
the cache breakpoint. Both are things a later edit can quietly undo without
any test failing anywhere else.
"""

from agent import (MODEL, REFUSAL_REPLY, MykolaAgent, _personalization,
                   _personalized_system, _stable_system, _system_blocks)


class _TextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, text="Good evening.", stop_reason="end_turn"):
        self.content = [_TextBlock(text)] if text else []
        self.stop_reason = stop_reason


class _Stream:
    """Enough of the SDK's streaming context manager for `answer()`."""

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
    def __init__(self, message):
        self._message = message
        self.calls = []

    def create(self, **kwargs):          # recap()
        self.calls.append(kwargs)
        return self._message

    def stream(self, **kwargs):          # answer()
        self.calls.append(kwargs)
        return _Stream(self._message)


class _Client:
    def __init__(self, message):
        self.messages = _Messages(message)


class _EmptyKB:
    """`answer()` retrieves before it asks; nothing needs to come back."""

    def retrieve(self, question, top_k=None):
        return []


def _agent(message):
    agent = MykolaAgent.__new__(MykolaAgent)
    agent.kb = _EmptyKB()
    agent.client = _Client(message)
    agent.card_saver = None
    agent.name_saver = None
    agent._cards_db = None
    return agent


def main() -> None:
    # --- the model -------------------------------------------------------
    assert MODEL == "claude-opus-5", MODEL

    # --- the prompt splits without changing ------------------------------
    # The one-string form is what the CLI and the persona tests read, so the
    # blocks must join back into exactly it — otherwise the cached path and
    # the tested path are two different prompts.
    for name, hidden in [(None, None), ("Anton", None), (None, ["Russian"]),
                         ("Anna Maria", ["Ukrainian", "Russian"]),
                         ("   ", ["Klingon"])]:
        blocks = _system_blocks(name, hidden, cache=True)
        assert "\n\n".join(b["text"] for b in blocks) == \
            _personalized_system(name, hidden), (name, hidden)
        assert "cache_control" in blocks[0], "the stable block carries the mark"
        assert all("cache_control" not in b for b in blocks[1:]), \
            "a breakpoint after the personalization would cache nothing shared"

    assert "cache_control" not in _system_blocks("Anton")[0], \
        "caching is opt-in: a prompt sent once is a write with no read"

    # The precondition for caching at all: the prefix is the same text for
    # everybody. If a visitor's name ever leaks into it, every learner
    # invalidates every other learner's entry.
    assert _stable_system() == _stable_system(), "the prefix must be stable"
    assert "Anton" not in _stable_system()
    assert "Anton" in _personalization("Anton")
    assert _personalization() == "", "nothing to say means no second block"
    # Roughly: 512 tokens is Opus 5's minimum cacheable prefix and English
    # runs well under 4 characters per token, so this is a safe offline proxy
    # for "long enough to cache" without spending an API call to count.
    assert len(_stable_system()) > 4 * 512, "too short to cache"

    # --- what each call site asks for ------------------------------------
    agent = _agent(_Message("A short recap."))
    agent.recap("user: hello\nMykola: good evening", user_name="Anton")
    (recap_call,) = agent.client.messages.calls
    # Opus 5 thinks when the parameter is omitted; RECAP_MAX_TOKENS caps
    # thinking and text together, so a thinking recap can arrive truncated —
    # or empty, which the site silently drops.
    assert recap_call["thinking"] == {"type": "disabled"}, recap_call["thinking"]
    assert recap_call["model"] == MODEL
    assert isinstance(recap_call["system"], str), \
        "the recap is one-shot; there is no second request to read a cache"

    agent = _agent(_Message("Good evening, Anton."))
    out = agent.answer("hello", user_name="Anton", hidden_languages=["Russian"])
    (answer_call,) = agent.client.messages.calls
    assert answer_call["thinking"] == {"type": "adaptive"}
    assert answer_call["model"] == MODEL
    system = answer_call["system"]
    assert isinstance(system, list) and "cache_control" in system[0], \
        "the chat path re-sends this prompt every turn; it should be cached"
    assert "Anton" not in system[0]["text"], "personalization sits after it"
    assert "Anton" in system[1]["text"]
    assert out["response"] == "Good evening, Anton."

    # --- a refusal is an answer, not a blank ------------------------------
    # Opus 5's classifiers decline with HTTP 200, stop_reason "refusal" and no
    # text. Unhandled that reaches the learner as an empty message.
    agent = _agent(_Message("", stop_reason="refusal"))
    assert agent.answer("...")["response"] == REFUSAL_REPLY

    agent = _agent(_Message("", stop_reason="refusal"))
    assert agent.recap("user: hello") == "", \
        "a refused recap is no recap; there is no question to answer instead"

    # A refusal after some text has already streamed keeps what was said.
    agent = _agent(_Message("Half an answer", stop_reason="refusal"))
    assert agent.answer("...")["response"] == "Half an answer"

    print("test_model_and_caching.py: all checks passed")


if __name__ == "__main__":
    main()
