# CLAUDE.md — ai_agent (Mykola)

Guidance for Claude Code (and contributors) working in this repo.

## What this is

**Mykola** — a RAG (retrieval-augmented) study assistant for English learning:
TF-IDF retrieval over a Markdown knowledge base, answered by Claude. He is a
persona-rich "distinguished gentleman" named after the composer Mykola
Leontovych. This repo is used two ways:

1. **Imported by [KuantorFlow](https://github.com/Kuantor/kuantorflow)** as its
   chat widget — the agent code is imported, **never duplicated**.
2. As its own **standalone Flask app** (`flask_app.py`).

## Core modules

- **`agent.py`** — the importable `MykolaAgent` class (retrieval + Claude call
  + the `TOOLS` list: `add_flashcard`, `set_preferred_name`), the
  `api_error_response` helper, `SYSTEM_PROMPT` (the persona/voice), and a CLI
  (`python agent.py`).
- **`rag.py`** — `KnowledgeBase` (TF-IDF over `knowledge/*.md`, heading-split).
- **`knowledge/*.md`** — documents that *back* the persona (British English,
  French connections, music), so its signature moves are retrieved and cited,
  not improvised.
- **`flask_app.py`** — the standalone web chat.

## The importable contract (keep it stable)

KuantorFlow calls `MykolaAgent`:

- `answer(question, history, on_text=None, user_name=None, hidden_languages=None)`
  → `{response, sources, history, saved_cards}`
- `recap(past_conversations, user_name=None, hidden_languages=None, away_hours=None)`
  — with `away_hours` (#54) the site has just restarted a stale conversation,
  so the recap opens by acknowledging the break; `build_recap_prompt()` and
  `describe_gap()` are module level so the wording is checkable offline.
- `__init__(card_saver=…, name_saver=…)` — KuantorFlow **injects the callables
  that touch its database**: `save_flashcard` so Mykola saves through the
  site's one write path (standalone falls back to `FlashcardsDB`), and a writer
  for `users.preferred_name` (#62), which has **no standalone fallback** —
  this repo has no notion of an account, so without a host the tool says it
  cannot remember the name.

**Adding a tool**: define it next to `ADD_FLASHCARD_TOOL`, add it to `TOOLS`,
give it a `_run_…` handler returning a **JSON status string** (never raising —
the model relays errors in character), and register it in `_run_tool()`. A
saver that refuses does so by raising; the handler turns that into an error
status. Say in `SYSTEM_PROMPT` when to reach for it, or the model won't.

New optional params are **feature-detected** by the caller
(`inspect.signature`), so the two repos can deploy in any order. Preserve that:
add new capability as optional kwargs, don't break the signatures above.

Mykola knows he is **Claude-powered** and admits it in character (#48); a
symbolic birthday drives his "age" answers.

## The model and the system prompt (#63/#64)

`MODEL = "claude-opus-5"`. Two things about it are easy to get wrong:

- **Thinking is on when you omit the parameter.** On Opus 4.8 omitting it
  meant *no* thinking. `answer()` asks for `{"type": "adaptive"}` and always
  did; `recap()` passes `{"type": "disabled"}` **on purpose** — its
  `max_tokens` is 1024 and that ceiling covers thinking and text together, so
  a thinking recap can come back truncated or empty, and an empty recap is
  silently dropped by the site.
- **A refusal is an HTTP 200**, `stop_reason == "refusal"` with no text. Both
  call sites check for it: `answer()` substitutes `REFUSAL_REPLY` so the
  learner is not left with a blank message, `recap()` returns `""`.
- **He is told which Claude he is.** `_model_guidance()` renders `MODEL`
  through `_model_label()` (`claude-opus-5` → "Claude Opus 5") into the prompt,
  so changing the constant changes what he answers — he can never name a model
  he is not running on. Without it he hedges: he knows he is Claude (#48) but
  nothing in a conversation tells him the version, so he says he cannot tell.
  The same block forbids guessing at training dates, benchmarks, or
  comparisons, which he equally cannot see.

The system prompt is built in three pieces so it can be **prompt-cached**:
`_stable_system()` (shared by everyone — the cached prefix),
`_personalization()` (name and hidden languages — must stay *after* the
breakpoint), and `_system_blocks(..., cache=True)` which assembles them.
`_personalized_system()` still returns the whole thing as one string for the
CLI and the tests. Caching is on for `answer()` only: a cached prefix costs
1.25× to write and 0.1× to read, so it pays off across the turns of a chat but
not for a one-shot recap. **Anything you add before the breakpoint must render
identically across a conversation** — check `usage.cache_read_input_tokens` is
non-zero on the second message if you touch it.

## Run & test

```bash
python agent.py                 # interactive CLI
python test_agent_prompt.py     # persona / prompt / knowledge-base checks
python test_preferred_name.py   # the set_preferred_name tool (#62)
python test_model_and_caching.py  # thinking mode, cache breakpoint, refusals
python test_rag.py
```

Tests are plain scripts (not pytest) that assert `SYSTEM_PROMPT` content and
knowledge-base retrieval — update them when you change the persona.

## Conventions

- **`ANTHROPIC_API_KEY`** lives in this repo's own `.env` (gitignored),
  separate from KuantorFlow's `.env`. Own venv.
- **Never commit secrets.** Never duplicate this code into KuantorFlow.
- Copyright guardrail: reference songs by title/theme, never full lyrics.
