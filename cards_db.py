"""Flashcards data access for Mykola.

Supported sources (in priority order):
- MySQL table (for KuantorFlow integration)
- SQLite table
- Local JSON fallback for standalone demos
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # Optional unless MySQL mode is configured.
    pymysql = None
    DictCursor = None


class FlashcardsDB:
    """Flashcards store with MySQL/SQLite/JSON backends."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize DB source from env, with JSON fallback.

        Environment options:
        - ``FLASHCARDS_DB_URL``: ``mysql://user:pass@host:3306/db``
        - ``FLASHCARDS_DB_SQLITE_PATH``: absolute/relative sqlite file path
        - ``FLASHCARDS_DB_URL``: ``sqlite:///...`` URL
        - ``FLASHCARDS_MYSQL_HOST``/``PORT``/``USER``/``PASSWORD``/``DATABASE``
        - ``FLASHCARDS_DB_TABLE``: optional table name (default: ``flashcards``)
        """
        self.table_name = (os.getenv("FLASHCARDS_DB_TABLE", "flashcards").strip() or "flashcards")
        self._validate_table_name(self.table_name)

        self.mysql_config = self._resolve_mysql_config()
        self.sqlite_path = self._resolve_sqlite_path()
        self.db_path = db_path or (Path(__file__).parent / "data" / "flashcards.json")
        self.cards: List[Dict] = []

        if self.mysql_config:
            if not self._mysql_has_table(self.table_name):
                raise RuntimeError(
                    f"MySQL DB configured but table '{self.table_name}' was not found in "
                    f"database '{self.mysql_config['database']}'."
                )
        elif self.sqlite_path:
            if not self._sqlite_has_table(self.table_name):
                raise RuntimeError(
                    f"SQLite DB configured at '{self.sqlite_path}' but table '{self.table_name}' was not found."
                )
        else:
            self.load_or_create()

    @staticmethod
    def _validate_table_name(name: str) -> None:
        """Allow simple SQL identifiers only to avoid SQL injection via env."""
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise RuntimeError("FLASHCARDS_DB_TABLE must be a simple SQL identifier.")

    def _resolve_mysql_config(self) -> Optional[Dict[str, Any]]:
        """Resolve MySQL connection config from URL or separate env vars."""
        db_url = os.getenv("FLASHCARDS_DB_URL", "").strip()
        if db_url.startswith("mysql://") or db_url.startswith("mysql+pymysql://"):
            if pymysql is None:
                raise RuntimeError(
                    "MySQL mode configured but 'pymysql' is not installed. "
                    "Install it and restart the app."
                )

            parsed = urlparse(db_url.replace("mysql+pymysql://", "mysql://", 1))
            db_name = (parsed.path or "").lstrip("/")
            if not db_name:
                raise RuntimeError("MySQL URL must include database name, e.g. mysql://user:pass@host/db")
            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 3306,
                "user": parsed.username,
                "password": parsed.password or "",
                "database": db_name,
            }

        mysql_host = os.getenv("FLASHCARDS_MYSQL_HOST", "").strip()
        if mysql_host:
            if pymysql is None:
                raise RuntimeError(
                    "MySQL mode configured but 'pymysql' is not installed. "
                    "Install it and restart the app."
                )
            return {
                "host": mysql_host,
                "port": int(os.getenv("FLASHCARDS_MYSQL_PORT", "3306")),
                "user": os.getenv("FLASHCARDS_MYSQL_USER", "").strip(),
                "password": os.getenv("FLASHCARDS_MYSQL_PASSWORD", ""),
                "database": os.getenv("FLASHCARDS_MYSQL_DATABASE", "").strip(),
            }
        return None

    def _resolve_sqlite_path(self) -> Optional[Path]:
        """Resolve sqlite path from env vars, if configured."""
        explicit = os.getenv("FLASHCARDS_DB_SQLITE_PATH", "").strip()
        if explicit:
            return Path(explicit).expanduser()

        url = os.getenv("FLASHCARDS_DB_URL", "").strip()
        if url.startswith("sqlite:///"):
            return Path(url.replace("sqlite:///", "", 1)).expanduser()
        return None

    def _mysql_connect(self):
        """Open a MySQL connection using current config."""
        return pymysql.connect(
            host=self.mysql_config["host"],
            port=self.mysql_config["port"],
            user=self.mysql_config["user"],
            password=self.mysql_config["password"],
            database=self.mysql_config["database"],
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
        )

    def _mysql_has_table(self, table_name: str) -> bool:
        """Check if target MySQL DB has a table."""
        try:
            with self._mysql_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = %s
                        LIMIT 1
                        """,
                        (self.mysql_config["database"], table_name),
                    )
                    return cur.fetchone() is not None
        except Exception:
            return False

    def _mysql_columns(self) -> List[str]:
        """Fetch flashcards table columns from MySQL."""
        with self._mysql_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SHOW COLUMNS FROM `{self.table_name}`")
                rows = cur.fetchall()
        return [r["Field"] for r in rows]

    def _sqlite_has_table(self, table_name: str) -> bool:
        """Check if target sqlite DB has a table."""
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    def _sqlite_columns(self) -> List[str]:
        """Fetch flashcards table columns from sqlite."""
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()
        return [r[1] for r in rows]

    @staticmethod
    def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
        """Choose the first matching candidate column (case-insensitive)."""
        lookup = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lookup:
                return lookup[cand.lower()]
        return None

    def _normalize_row(self, data: Dict[str, Any], columns: List[str]) -> Dict:
        """Map arbitrary flashcards schema to app-normalized card dict."""
        word_col = self._pick_column(columns, ["word", "term", "english", "front", "english_word", "word_en"])
        translation_col = self._pick_column(
            columns,
            ["translation", "meaning", "back", "native", "ukrainian", "russian", "translation_uk", "translation_ru"],
        )
        explanation_col = self._pick_column(columns, ["explanation", "example", "description", "note", "notes"])
        category_col = self._pick_column(columns, ["category", "topic", "tag", "part_of_speech", "pos"])
        id_col = self._pick_column(columns, ["id", "card_id"])

        return {
            "id": data.get(id_col) if id_col else None,
            "word": str(data.get(word_col, "")).strip() if word_col else "",
            "translation": str(data.get(translation_col, "")).strip() if translation_col else "",
            "explanation": str(data.get(explanation_col, "")).strip() if explanation_col else "",
            "category": str(data.get(category_col, "general")).strip() if category_col else "general",
        }

    def _query_mysql_cards(self, where: str = "", params: tuple = ()) -> List[Dict]:
        """Run a cards query against MySQL and normalize rows."""
        columns = self._mysql_columns()
        query = f"SELECT * FROM `{self.table_name}`"
        if where:
            query += f" WHERE {where}"
        with self._mysql_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        cards = [self._normalize_row(r, columns) for r in rows]
        return [c for c in cards if c.get("word")]

    def _query_sqlite_cards(self, where: str = "", params: tuple = ()) -> List[Dict]:
        """Run a cards query against sqlite and normalize rows."""
        columns = self._sqlite_columns()
        query = f"SELECT * FROM {self.table_name}"
        if where:
            query += f" WHERE {where}"
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        cards = [self._normalize_row(dict(r), columns) for r in rows]
        return [c for c in cards if c.get("word")]

    def load_or_create(self):
        """Load database from file or create with sample data if it doesn't exist."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.cards = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cards = self._create_sample_cards()
                self.save()
        else:
            self.cards = self._create_sample_cards()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.save()

    def save(self):
        """Save the database to file."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.cards, f, ensure_ascii=False, indent=2)

    # Full kuantorflow flashcards structure (issue #20).
    FULL_CARD_FIELDS = (
        "word", "pos", "explanation_en", "examples_en",
        "translation_ukr", "examples_ukr", "translation_rus", "examples_rus",
        "topic",
    )

    def add_full_flashcard(self, entry: Dict[str, Any]) -> Dict:
        """
        Insert one flashcard with the full kuantorflow column structure.
        Used by the Mykola agent's add_flashcard tool in standalone mode
        (embedded kuantorflow injects its own save_flashcard instead).

        Only columns that actually exist in the target table are written, so
        this works against the real kuantorflow schema as well as simplified
        ones. Returns the saved card dict (with id where the backend has one).
        """
        word = str(entry.get("word") or "").strip()
        if not word:
            raise ValueError("word is required")

        if self.mysql_config or self.sqlite_path:
            columns = self._mysql_columns() if self.mysql_config else self._sqlite_columns()
            lookup = {c.lower(): c for c in columns}
            data = {}
            for field in self.FULL_CARD_FIELDS:
                value = entry.get(field)
                if value and field.lower() in lookup:
                    data[lookup[field.lower()]] = str(value)
            if not data:
                raise RuntimeError(
                    f"Table '{self.table_name}' has none of the flashcard columns."
                )
            cols = list(data.keys())
            values = tuple(data[c] for c in cols)
            if self.mysql_config:
                col_sql = ", ".join(f"`{c}`" for c in cols)
                placeholders = ", ".join(["%s"] * len(cols))
                with self._mysql_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"INSERT INTO `{self.table_name}` ({col_sql}) VALUES ({placeholders})",
                            values,
                        )
                        new_id = cur.lastrowid
            else:
                col_sql = ", ".join(cols)
                placeholders = ", ".join(["?"] * len(cols))
                with sqlite3.connect(self.sqlite_path) as conn:
                    cur = conn.execute(
                        f"INSERT INTO {self.table_name} ({col_sql}) VALUES ({placeholders})",
                        values,
                    )
                    conn.commit()
                    new_id = cur.lastrowid
            return {"id": new_id, **data}

        # JSON fallback: normalize to the demo card shape.
        card = {
            "id": len(self.cards) + 1,
            "word": word,
            "translation": str(entry.get("translation_ukr") or entry.get("translation_rus") or "").strip(),
            "explanation": str(entry.get("explanation_en") or "").strip(),
            "category": str(entry.get("topic") or "general").strip(),
        }
        self.cards.append(card)
        self.save()
        return card

    def add_card(self, word: str, translation: str, explanation: str, category: str = "general") -> Dict:
        """
        Add a new flashcard to the database.
        Returns the created card dict.
        """
        if self.mysql_config or self.sqlite_path:
            raise RuntimeError(
                "Adding cards through this endpoint is disabled for external DBs; "
                "manage cards in KuantorFlow directly."
            )

        card = {
            "id": len(self.cards) + 1,
            "word": word,
            "translation": translation,
            "explanation": explanation,
            "category": category,
        }
        self.cards.append(card)
        self.save()
        return card

    def get_all_cards(self) -> List[Dict]:
        """Get all flashcards."""
        if self.mysql_config:
            return self._query_mysql_cards()
        if self.sqlite_path:
            return self._query_sqlite_cards()
        return self.cards

    def get_cards_by_category(self, category: str) -> List[Dict]:
        """Get flashcards by category."""
        if self.mysql_config:
            # Try direct category match first; if schema has no category, fallback to client-side filter.
            try:
                return self._query_mysql_cards("LOWER(category)=LOWER(%s)", (category,))
            except Exception:
                pass
            cards = self._query_mysql_cards()
            return [c for c in cards if c.get("category", "").lower() == category.lower()]
        if self.sqlite_path:
            # Try direct category match first; if schema has no category, fallback to client-side filter.
            try:
                return self._query_sqlite_cards("LOWER(category)=LOWER(?)", (category,))
            except sqlite3.Error:
                pass
            cards = self._query_sqlite_cards()
            return [c for c in cards if c.get("category", "").lower() == category.lower()]
        return [c for c in self.cards if c.get("category", "").lower() == category.lower()]

    def search_cards(self, query: str) -> List[Dict]:
        """Search cards by word or translation (partial match)."""
        q = query.lower()
        cards = self.get_all_cards()
        return [
            c
            for c in cards
            if q in c.get("word", "").lower()
            or q in c.get("translation", "").lower()
            or q in c.get("explanation", "").lower()
        ]

    def get_cards_count(self) -> int:
        """Get total number of cards."""
        if self.mysql_config:
            with self._mysql_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) AS c FROM `{self.table_name}`")
                    row = cur.fetchone()
                    return int((row or {}).get("c", 0))
        if self.sqlite_path:
            with sqlite3.connect(self.sqlite_path) as conn:
                row = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
                return int(row[0] if row else 0)
        return len(self.cards)

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        cards = self.get_all_cards()
        categories = set(c.get("category", "general") for c in cards)
        return sorted(list(categories))

    @staticmethod
    def _create_sample_cards() -> List[Dict]:
        """Create sample flashcards for demonstration."""
        return [
            {
                "id": 1,
                "word": "phrasal verb",
                "translation": "фразовий дієслово",
                "explanation": "A verb combined with a preposition or adverb that has a different meaning from the individual words",
                "category": "grammar"
            },
            {
                "id": 2,
                "word": "present perfect",
                "translation": "Present Perfect (теперішній дослідний час)",
                "explanation": "A tense used to describe actions that started in the past and continue to the present",
                "category": "grammar"
            },
            {
                "id": 3,
                "word": "irregular verb",
                "translation": "неправильне дієслово",
                "explanation": "A verb that does not follow the standard pattern of adding -ed for past tense",
                "category": "grammar"
            },
            {
                "id": 4,
                "word": "travel",
                "translation": "подорож, подорожувати",
                "explanation": "The act of going from one place to another, often over a long distance",
                "category": "vocabulary"
            },
            {
                "id": 5,
                "word": "itinerary",
                "translation": "маршрут, план подорожі",
                "explanation": "A planned route or sequence of travel",
                "category": "travel"
            },
            {
                "id": 6,
                "word": "accommodation",
                "translation": "житло, розміщення",
                "explanation": "A place where someone can live or stay temporarily",
                "category": "travel"
            },
            {
                "id": 7,
                "word": "dialect",
                "translation": "діалект",
                "explanation": "A particular form of a language that is spoken in a specific region",
                "category": "linguistics"
            },
            {
                "id": 8,
                "word": "eloquent",
                "translation": "красномовний",
                "explanation": "Fluent or persuasive in speaking or writing",
                "category": "vocabulary"
            }
        ]


def format_card_list_for_chat(cards: List[Dict]) -> str:
    """Format a list of flashcards for chat display."""
    if not cards:
        return "No cards found."
    
    output = "📇 **Flashcard Database**\n\n"
    for card in cards:
        output += f"**{card['word']}** ({card.get('category', 'general')})\n"
        output += f"  🇺🇦 {card['translation']}\n"
        output += f"  📝 {card['explanation']}\n\n"
    
    output += f"Total: {len(cards)} card(s)"
    return output


def format_card_count_for_chat(count: int, category: Optional[str] = None) -> str:
    """Format card count message for chat."""
    if category:
        return f"📇 Found {count} card(s) in the '{category}' category."
    return f"📇 The database has {count} flashcard(s) in total."
