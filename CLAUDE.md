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

- **`agent.py`** — the importable `MykolaAgent` class (retrieval + Claude
  call + `add_flashcard` tool), the `api_error_response` helper, `SYSTEM_PROMPT`
  (the persona/voice), and a CLI (`python agent.py`).
- **`rag.py`** — `KnowledgeBase` (TF-IDF over `knowledge/*.md`, heading-split).
- **`knowledge/*.md`** — documents that *back* the persona (British English,
  French connections, music), so its signature moves are retrieved and cited,
  not improvised.
- **`flask_app.py`** — the standalone web chat.

## The importable contract (keep it stable)

KuantorFlow calls `MykolaAgent`:

- `answer(question, history, on_text=None, user_name=None, hidden_languages=None)`
  → `{response, sources, history, saved_cards}`
- `recap(past_conversations, user_name=None, hidden_languages=None)`
- `__init__(card_saver=…)` — KuantorFlow **injects its `save_flashcard`** so
  Mykola saves through the site's one write path; standalone uses `FlashcardsDB`.

New optional params are **feature-detected** by the caller
(`inspect.signature`), so the two repos can deploy in any order. Preserve that:
add new capability as optional kwargs, don't break the signatures above.

Mykola knows he is **Claude-powered** and admits it in character (#48); a
symbolic birthday drives his "age" answers.

## Run & test

```bash
python agent.py                 # interactive CLI
python test_agent_prompt.py     # persona / prompt / knowledge-base checks
python test_rag.py
```

Tests are plain scripts (not pytest) that assert `SYSTEM_PROMPT` content and
knowledge-base retrieval — update them when you change the persona.

## Conventions

- **`ANTHROPIC_API_KEY`** lives in this repo's own `.env` (gitignored),
  separate from KuantorFlow's `.env`. Own venv.
- **Never commit secrets.** Never duplicate this code into KuantorFlow.
- Copyright guardrail: reference songs by title/theme, never full lyrics.
