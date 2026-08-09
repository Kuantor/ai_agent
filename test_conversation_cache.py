"""Caching the conversation prefix (#71).

Offline: everything here is about the shape of the request we would send, so
no API key and no network. The one thing these checks cannot prove is that the
cache actually *hits* — that is `cache_read_input_tokens` on a live call, which
is exactly why #71 also added the usage log.
"""

import logging

import agent as A

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


def marker(msg):
    """The cache_control on a message's last block, or None."""
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return None
    last = content[-1]
    return last.get("cache_control") if isinstance(last, dict) else None


print("the breakpoint lands on the end of the conversation")
convo = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Good day."},
    {"role": "user", "content": "tell me about groggy"},
]
out = A._cache_conversation(convo)
check("the last message is marked", marker(out[-1]) == {"type": "ephemeral"})
check("no earlier message is", all(marker(m) is None for m in out[:-1]))
check("a string content becomes a block",
      out[-1]["content"] == [{"type": "text", "text": "tell me about groggy",
                              "cache_control": {"type": "ephemeral"}}])
check("exactly one breakpoint is added",
      sum(marker(m) is not None for m in out) == 1)

print("\nit does not mutate the caller's history")
check("the input list is untouched",
      convo[-1]["content"] == "tell me about groggy")
check("...which matters because history goes back to the browser",
      all(isinstance(m["content"], str) for m in convo))

print("\nblock content")
blocks = [{"role": "user", "content": [
    {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]
out = A._cache_conversation(blocks)
check("marks the last block, not the first",
      out[0]["content"][1].get("cache_control") and
      "cache_control" not in out[0]["content"][0])
check("the original blocks are not mutated",
      "cache_control" not in blocks[0]["content"][1])


class SdkBlock:
    """A block object the SDK returned — not a dict, cannot take the marker."""
    type = "tool_use"


odd = [{"role": "assistant", "content": [SdkBlock()]}]
check("an SDK block object is left alone rather than crashing",
      A._cache_conversation(odd) == odd)
check("an empty conversation is returned as-is", A._cache_conversation([]) == [])
check("unknown content is left alone",
      A._cache_conversation([{"role": "user", "content": 7}])[0]["content"] == 7)

print("\n#64's breakpoint is still where it was")
sys_blocks = A._system_blocks("Anton", None, cache=True)
check("the stable system block carries the marker",
      sys_blocks[0].get("cache_control") == {"type": "ephemeral"})
check("still exactly one system breakpoint",
      sum("cache_control" in b for b in sys_blocks) == 1)
check("two breakpoints in total, well under the limit of four",
      sum("cache_control" in b for b in sys_blocks) +
      sum(marker(m) is not None for m in A._cache_conversation(convo)) == 2)
check("personalization stays after the breakpoint, so it never invalidates it",
      len(sys_blocks) == 1 or "cache_control" not in sys_blocks[-1])


class Usage:
    input_tokens, output_tokens = 120, 300
    cache_creation_input_tokens, cache_read_input_tokens = 0, 3368


class Msg:
    usage, stop_reason = Usage(), "end_turn"


print("\nusage logging")
records = []
handler = logging.Handler()
handler.emit = records.append
log = logging.getLogger("mykola.usage")
log.addHandler(handler)
log.setLevel(logging.INFO)

A._log_usage(Msg())
check("a call is logged", len(records) == 1)
check("it carries the cache figures",
      "cache_read=3368" in records[0].getMessage() and
      "in=120" in records[0].getMessage())

A._log_usage(object())          # no usage attribute
A._log_usage(None)              # nothing at all
check("a message with no usage logs nothing and does not raise",
      len(records) == 1)

print(f"\n{ok} checks passed")
