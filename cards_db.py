"""
Flashcards database for Tynna.
Manages and retrieves flashcards (words/expressions with translations and explanations).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class FlashcardsDB:
    """
    Simple JSON-based flashcards database.
    Stores cards with word, translation, explanation, and category.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the flashcards database."""
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "flashcards.json"
        
        self.db_path = db_path
        self.cards = []
        self.load_or_create()

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

    def add_card(self, word: str, translation: str, explanation: str, category: str = "general") -> Dict:
        """
        Add a new flashcard to the database.
        Returns the created card dict.
        """
        card = {
            "id": len(self.cards) + 1,
            "word": word,
            "translation": translation,
            "explanation": explanation,
            "category": category
        }
        self.cards.append(card)
        self.save()
        return card

    def get_all_cards(self) -> List[Dict]:
        """Get all flashcards."""
        return self.cards

    def get_cards_by_category(self, category: str) -> List[Dict]:
        """Get flashcards by category."""
        return [c for c in self.cards if c.get("category", "").lower() == category.lower()]

    def search_cards(self, query: str) -> List[Dict]:
        """Search cards by word or translation (partial match)."""
        query = query.lower()
        results = [
            c for c in self.cards
            if query in c.get("word", "").lower() or query in c.get("translation", "").lower()
        ]
        return results

    def get_cards_count(self) -> int:
        """Get total number of cards."""
        return len(self.cards)

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        categories = set(c.get("category", "general") for c in self.cards)
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
