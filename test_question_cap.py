"""The standalone app caps one question's length (kuantorflow#564).

`MAX_CONTENT_LENGTH` bounds the question and the history together, so on its
own it let one pasted message of close to a megabyte reach the model as a
single message. `/api/chat` now refuses a question longer than
`agent.MAX_QUESTION_CHARS`, before the agent is asked anything, and the page's
input carries the same number as `maxlength`.

Offline: the agent, the card database and the word list are replaced before
`flask_app` is imported, so nothing reaches the network or MySQL.

Run: python test_question_cap.py
"""

import importlib
import sys

import agent
import cards_db
import word_list

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


class _KB:
    chunks = []


class _Agent:
    chunk_count = 0

    def __init__(self, *a, **k):
        self.kb = _KB()
        self.questions = []

    def answer(self, question, history=None, **kwargs):
        self.questions.append(question)
        return {"response": "Indeed.", "sources": [], "history": [],
                "saved_cards": []}


class _Cards:
    def get_all_cards(self):
        return []

    def get_categories(self):
        return []


def _load_app():
    agent.MykolaAgent = _Agent
    cards_db.FlashcardsDB = _Cards
    word_list.WordListGenerator = lambda chunks: None
    sys.modules.pop("flask_app", None)
    module = importlib.import_module("flask_app")
    # The chat log is not what this file tests; keep it off the disk.
    module._append_chat_log = lambda *a, **k: None
    return module


def main():
    flask_app = _load_app()
    client = flask_app.app.test_client()
    limit = agent.MAX_QUESTION_CHARS

    print("the cap")
    check("it is 2,000, the same as KuantorFlow's default", limit == 2000)

    r = client.post("/api/chat", json={"question": "a" * limit})
    check("exactly the limit is answered", r.status_code == 200)
    check("...and reaches the agent", flask_app.agent.questions == ["a" * limit])

    r = client.post("/api/chat", json={"question": "a" * (limit + 1)})
    check("one character over is refused", r.status_code == 400)
    check("...with the sentence", r.get_json()["error"] ==
          "Please keep a message to Mykola under 2,000 characters.")
    check("...and never reaches the agent", len(flask_app.agent.questions) == 1)

    r = client.post("/api/chat", json={"question": "ї" * limit})
    check("characters, not bytes: 2,000 Ukrainian letters pass",
          r.status_code == 200)

    print("\nthe page")
    html = client.get("/").get_data(as_text=True)
    check("the input carries the same maxlength", f'maxlength="{limit}"' in html)

    print(f"\n{ok} checks passed")


if __name__ == "__main__":
    main()
