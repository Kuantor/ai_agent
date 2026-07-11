# ai_agent

A primitive RAG agent for [KuantorFlow](https://github.com/Kuantor/kuantorflow):
answers English-learning questions grounded in a local knowledge base, with
Claude (Anthropic API) as the language model.

## How it works

```
question ──> retrieval (TF-IDF over knowledge/*.md) ──> top-3 chunks
                                                            │
                    Claude (claude-opus-4-8) <── question + <context>
                          │
                       answer (streamed)
```

- **`knowledge/`** — the knowledge base: plain markdown files (grammar tips,
  irregular verbs, learning strategies, KuantorFlow FAQ). Add or edit files
  and the index rebuilds on the next start. No database needed.
- **`rag.py`** — splits documents into chunks at headings and finds the most
  relevant ones for a question (TF-IDF cosine similarity, in memory).
- **`agent.py`** — interactive chat: retrieves context, sends it with your
  question and the conversation history to Claude, streams the answer.

## Setup

```powershell
cd C:\Users\38050\Documents\!Projects\ai_agent
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
copy .env.example .env    # then put your real API key into .env
```

An API key comes from https://console.anthropic.com (API Keys section);
usage is pay-per-token.

## Running

```powershell
.\venv\Scripts\python test_rag.py   # offline check of the retrieval layer
.\venv\Scripts\python agent.py      # interactive chat (needs the API key)
```

Example session:

```
you> when do I use present perfect?
[retrieved: grammar_tips.md#Present Perfect vs Past Simple (0.42), ...]
agent> Use Present Perfect when the time is not stated or the action ...
```

## Next step

The plan is to embed this agent into KuantorFlow so users can ask
English-learning questions from the website. The `KnowledgeBase` and the
prompt-building logic are plain importable modules, so the Flask app can
call them directly; the knowledge base can later be extended with content
generated from the flashcards database.

## Error handling: token exhaustion

When the Anthropic / Claude account associated with the agent runs out of
tokens or credits, the API returns an error. To make this situation clearer
to users, the agent now provides friendly, actionable messages both in the
console and via the web UI:

- Console (`agent.py`): prints a short message:
  - "You are out of Claude tokens (insufficient Anthropic credits)."
  - It also shows the billing URL where you can top up: `https://console.anthropic.com/account/billing/overview`.
  - If the API response includes retry/reset headers (for example
    `Retry-After` or `X-Rate-Limit-Reset`), a short "Try again in Xm Ys."
    hint is appended to the console message.

- Web API (`/api/chat` in `flask_app.py`): returns HTTP `402 Payment Required`
  with JSON in the form:

  ```json
  {
    "error": "You are out of Claude tokens (insufficient Anthropic credits). Please top up your account at: https://console.anthropic.com/account/billing/overview.",
    "retry_in_seconds": 123 // optional, present only if the API provided reset info
  }
  ```

  The `retry_in_seconds` field is optional and will be present only when
  the Anthropic error response includes a retry or reset header that can be
  parsed into a number of seconds.

Notes:
- The message wording is intentionally concise and user-facing; it avoids
  exposing raw API internals while still directing users to the billing
  console.
- If you prefer different wording (e.g., removing the name "Claude" or
  localizing the message), update `agent.py` and `flask_app.py` accordingly.
