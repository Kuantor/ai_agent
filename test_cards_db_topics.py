"""A card's topic comes from the topics table, not the flashcards.topic string (#67).

kuantorflow#207 gave topics their own table and pointed `flashcards.topic_id`
at it; the old `topic` string is still written only until kuantorflow drops it
(kuantorflow#210). `FlashcardsDB` used to find a card's topic by sniffing for a
column, and with `topic` gone the next candidate it would have found is `pos`:
every card's part of speech reported as its topic, with no error anywhere.

So, in order:
  1. `topics` joined through `flashcards.topic_id`, when both are there;
  2. a plain `category`, `topic` or `tag` column, for simplified schemas;
  3. "general", for a table with none of them -- never `pos`.

Embedded in kuantorflow none of this runs (the host injects its own readers,
#68); it is the standalone app and CLI, which a developer may point at
kuantorflow's database. Plain script like the other tests here: sqlite files in
a temp folder, no API key, no MySQL.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

for name in ("FLASHCARDS_DB_URL", "FLASHCARDS_MYSQL_HOST", "FLASHCARDS_DB_SQLITE_PATH",
             "FLASHCARDS_DB_TABLE"):
    os.environ.pop(name, None)

import cards_db
from cards_db import FlashcardsDB

ok = 0


def check(label, condition):
    global ok
    assert condition, f"FAILED: {label}"
    ok += 1
    print(f"  ok  {label}")


TMP = Path(tempfile.mkdtemp(prefix="cards_db_topics_"))

KF_TOPICS = """
CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                     namespace INTEGER NOT NULL DEFAULT 0);
INSERT INTO topics (id, name) VALUES (1, 'Work and careers'), (2, 'Empty topic'),
                                     (3, 'Daily life');
-- A learner's private topic lives in their own namespace (kuantorflow#382).
INSERT INTO topics (id, name, namespace) VALUES (4, 'My private notes', 7);
"""


def _db(name, script):
    path = TMP / f"{name}.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(script)
    return path


def _open(path):
    os.environ["FLASHCARDS_DB_SQLITE_PATH"] = str(path)
    try:
        return FlashcardsDB()
    finally:
        os.environ.pop("FLASHCARDS_DB_SQLITE_PATH", None)


# kuantorflow today: the string is still written, but it can be stale -- the
# card's topic is whatever topic_id points at.
WITH_COLUMN = _db("with_column", KF_TOPICS + """
CREATE TABLE flashcards (id INTEGER PRIMARY KEY, word TEXT, pos TEXT,
    explanation_en TEXT, translation_ukr TEXT, topic TEXT, topic_id INTEGER);
INSERT INTO flashcards (word, pos, translation_ukr, topic, topic_id) VALUES
    ('resign', 'verb', 'звільнитися', 'STALE NAME', 1),
    ('commute', 'verb', 'їздити на роботу', 'Daily life', 3),
    ('loose', 'adjective', 'вільний', NULL, NULL);
""")

# kuantorflow after #210: no string at all.
DROPPED = _db("dropped", KF_TOPICS + """
CREATE TABLE flashcards (id INTEGER PRIMARY KEY, word TEXT, pos TEXT,
    explanation_en TEXT, translation_ukr TEXT, topic_id INTEGER);
INSERT INTO flashcards (word, pos, translation_ukr, topic_id) VALUES
    ('resign', 'verb', 'звільнитися', 1),
    ('commute', 'verb', 'їздити на роботу', 3),
    ('loose', 'adjective', 'вільний', NULL);
""")

SIMPLE_TOPIC = _db("simple_topic", """
CREATE TABLE flashcards (id INTEGER PRIMARY KEY, word TEXT, translation TEXT, topic TEXT);
INSERT INTO flashcards (word, translation, topic) VALUES
    ('resign', 'звільнитися', 'Work'), ('commute', 'їздити', 'Travel');
""")

SIMPLE_CATEGORY = _db("simple_category", """
CREATE TABLE flashcards (id INTEGER PRIMARY KEY, word TEXT, translation TEXT, category TEXT);
INSERT INTO flashcards (word, translation, category) VALUES
    ('resign', 'звільнитися', 'Work'), ('commute', 'їздити', 'Travel');
""")

# Neither a topic nor a category: the part of speech must not stand in for one.
POS_ONLY = _db("pos_only", """
CREATE TABLE flashcards (id INTEGER PRIMARY KEY, word TEXT, translation TEXT,
                         pos TEXT, part_of_speech TEXT);
INSERT INTO flashcards (word, translation, pos, part_of_speech) VALUES
    ('resign', 'звільнитися', 'verb', 'verb');
""")


def _by_word(cards):
    return {c["word"]: c for c in cards}


print("kuantorflow's schema: the topic comes through topic_id")
for label, path in (("column still there", WITH_COLUMN), ("column dropped", DROPPED)):
    db = _open(path)
    cards = _by_word(db.get_all_cards())
    check(f"{label}: the name is the topics row's, not the string",
          cards["resign"]["category"] == "Work and careers")
    check(f"{label}: a card with no topic still comes back, with no category",
          "loose" in cards and cards["loose"]["category"] == "")
    check(f"{label}: one topic's cards, matched case-insensitively",
          [c["word"] for c in db.get_cards_by_category("work AND careers")] == ["resign"])
    check(f"{label}: categories are the topics that have cards, and only those",
          db.get_categories() == ["Daily life", "Work and careers"])


print("filtering happens in SQL, not by scanning every card")
calls = []
real = FlashcardsDB._query_sqlite_cards


def spy(self, where="", params=()):
    calls.append(where)
    return real(self, where, params)


FlashcardsDB._query_sqlite_cards = spy
try:
    for path in (WITH_COLUMN, DROPPED, SIMPLE_TOPIC, SIMPLE_CATEGORY):
        calls.clear()
        _open(path).get_cards_by_category("Work")
        check(f"{path.stem}: one query, with a WHERE clause",
              len(calls) == 1 and calls[0].strip() != "")
finally:
    FlashcardsDB._query_sqlite_cards = real


print("simplified schemas keep working")
for path, column in ((SIMPLE_TOPIC, "topic"), (SIMPLE_CATEGORY, "category")):
    db = _open(path)
    check(f"a plain {column} column is the topic",
          _by_word(db.get_all_cards())["resign"]["category"] == "Work")
    check(f"a plain {column} column filters",
          [c["word"] for c in db.get_cards_by_category("work")] == ["resign"])
    check(f"a plain {column} column lists its values",
          db.get_categories() == ["Travel", "Work"])


print("the part of speech is never a topic")
db = _open(POS_ONLY)
check("a table with pos and part_of_speech but no topic files the card under general",
      db.get_all_cards()[0]["category"] == "general")
check("and lists general as its only category", db.get_categories() == ["general"])
check("and asking for the part of speech finds nothing",
      db.get_cards_by_category("verb") == [])


print("the JSON demo is unchanged")
demo = FlashcardsDB(db_path=TMP / "demo.json")
check("the demo lists its sample categories",
      demo.get_categories() == sorted({c["category"] for c in demo.cards}))
check("and filters by one", all(c["category"] == "travel"
                                for c in demo.get_cards_by_category("travel"))
      and demo.get_cards_by_category("travel"))


print("standalone writes file a card under an existing topic, and never invent one")
for label, path in (("column still there", WITH_COLUMN), ("column dropped", DROPPED)):
    db = _open(path)
    db.add_full_flashcard({"word": "promotion", "pos": "noun", "topic": "Work and careers"})
    db.add_full_flashcard({"word": "zeal", "pos": "noun", "topic": "A topic nobody made"})
    db.add_full_flashcard({"word": "secret", "pos": "noun", "topic": "My private notes"})
    with sqlite3.connect(path) as conn:
        ids = dict(conn.execute("SELECT word, topic_id FROM flashcards").fetchall())
        names = [r[0] for r in conn.execute("SELECT name FROM topics ORDER BY id")]
    check(f"{label}: an existing topic's name becomes its id", ids["promotion"] == 1)
    check(f"{label}: an unknown name leaves the card without a topic", ids["zeal"] is None)
    check(f"{label}: somebody's private topic is never borrowed", ids["secret"] is None)
    check(f"{label}: and no topics row is created", "A topic nobody made" not in names)
    check(f"{label}: the new card reads back under its topic",
          "promotion" in [c["word"] for c in db.get_cards_by_category("Work and careers")])


print("the tool the model sees still speaks names")
import agent as A

props = A.ADD_FLASHCARD_TOOL["input_schema"]["properties"]
check("add_flashcard takes a topic name, a string", props["topic"]["type"] == "string")
check("and no id of any kind", not any("id" in key.split("_") for key in props))

print(f"\n{ok} checks passed")
