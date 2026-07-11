"""
Retrieval part of the RAG agent.

Loads markdown documents from the knowledge/ folder, splits them into
chunks, and finds the chunks most relevant to a question using TF-IDF
cosine similarity — no database and no external service needed.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")


@dataclass
class Chunk:
    source: str   # file the chunk came from
    heading: str  # section heading, if any
    text: str
    score: float = 0.0


def _split_into_chunks(source: str, text: str) -> list[Chunk]:
    """
    Split a markdown document into chunks at headings (#, ##, ...).
    Text before the first heading becomes its own chunk.
    """
    chunks = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(Chunk(source=source, heading=current_heading, text=body))

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            flush()
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


class KnowledgeBase:
    """In-memory TF-IDF index over the knowledge/ folder."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.chunks: list[Chunk] = []
        for path in sorted(knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            self.chunks.extend(_split_into_chunks(path.name, text))
        if not self.chunks:
            raise RuntimeError(f"No knowledge documents found in {knowledge_dir}")

        # Heading words matter for matching, so index them with the body.
        corpus = [f"{c.heading}\n{c.text}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[Chunk]:
        """Return the top_k most relevant chunks for the query."""
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] < min_score:
                break
            chunk = self.chunks[i]
            results.append(Chunk(chunk.source, chunk.heading, chunk.text, float(scores[i])))
        return results
