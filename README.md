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

## UI: header integration (Issue #7)

The chat header has been integrated into the main site header so the
conversation begins directly beneath the unified header. See
`templates/base.html` and `templates/index.html` for the layout changes.

## Word Lists and Gap Exercises (Issue #11)

Tynna can now generate vocabulary word lists and fill-the-gaps exercises from
the knowledge base. These features help users practice and reinforce their
English learning.

### Features:

- **Word List Generator** (`word_list.py`): Extracts key terms and phrases from
  the knowledge base, with definitions and sources.
- **Gap Exercises**: Creates fill-the-gaps exercises from knowledge base content
  where users must guess missing words.
- **API Endpoints**:
  - `POST /api/word-list` — generates a vocabulary list
  - `GET /api/gap-exercise` — generates a fill-the-gaps exercise

### Usage:

Users can ask Tynna for vocabulary lists or practice exercises, and the agent
will recognize these requests and suggest using the word list or gap exercise
features. The system prompt includes guidance for handling such requests.

Example:
```
user> I want a vocabulary list
tynna> I can generate a vocabulary list from the learning database! 
       Check the word list feature in the app.
```

### Flashcards Database:

Tynna can retrieve flashcards via `cards_db.py` from either:
- a real KuantorFlow MySQL table `flashcards` (preferred), or
- a SQLite table `flashcards`, or
- local JSON fallback (`data/flashcards.json`) for standalone demos.

To connect to real KuantorFlow MySQL in `.env`, use either:
- `FLASHCARDS_DB_URL=mysql://user:password@host:3306/kuantorflow`
or separate vars:
- `FLASHCARDS_MYSQL_HOST=127.0.0.1`
- `FLASHCARDS_MYSQL_PORT=3306`
- `FLASHCARDS_MYSQL_USER=your_user`
- `FLASHCARDS_MYSQL_PASSWORD=your_password`
- `FLASHCARDS_MYSQL_DATABASE=kuantorflow`
- `FLASHCARDS_DB_TABLE=flashcards`

SQLite fallback example:
- `FLASHCARDS_DB_SQLITE_PATH=C:/path/to/kuantorflow.db`
- `FLASHCARDS_DB_URL=sqlite:///C:/path/to/kuantorflow.db`

**Features:**
- `GET /api/cards` — retrieve all flashcards
- `GET /api/cards/category/<category>` — get cards from a specific category (e.g., "grammar", "vocabulary", "travel")
- `GET /api/cards/search?q=<query>` — search cards by word or translation
- `POST /api/cards/add` — add a new flashcard (requires: word, translation; optional: explanation, category)

**Tynna's Capabilities:**
When a user asks for "a list of all the cards in the database", "show me the list of words", or similar, Tynna now:
- She can retrieve the full card list and explain how to browse by category
- She can help search for specific words
- She can guide users to add new cards to the database
- She recognizes requests like "cards about travel" and can suggest filtering by category

Example with new database:
```
user> Can you give me a list of all the cards in the database?
tynna> Of course! The database has 8 flashcards across several categories: 
       grammar, linguistics, travel, and vocabulary. 
       
       You can:
       - View all cards
       - Browse by category (travel, grammar, vocabulary, etc.)
       - Search for specific words
       
       What cards would you like to explore?
```

## Dialog Logging (Issue #23)

Every web chat dialog with Tynna is logged to the repository directory:

- `tynna_logs/`

File naming format:

- `chat_<chat_id>.txt`

Each file contains timestamped user/assistant exchanges for that dialog.

## Persona Tone (Issue #10)

Tynna's conversational style was refined to feel warmer, more feminine, and
more personable while staying appropriate for a learning assistant.

Key adjustments:
- More empathetic, supportive, and graceful phrasing.
- Light playful charm to keep conversations engaging.
- Strict professional boundaries: no explicit or inappropriate flirting.
- Consistent tone across both console and browser chat views (shared
  `SYSTEM_PROMPT` in `agent.py`).

Rationale:
- Improve approachability and user comfort.
- Preserve clarity and educational value as the top priority.
