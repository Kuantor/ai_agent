"""
Word list and gap-filling exercise generator for Tynna.
Provides functionality to extract word lists from the knowledge base,
generate explanations/translations, and create fill-the-gaps exercises.
"""

import re
from typing import List, Dict, Tuple
from rag import Chunk


class WordListGenerator:
    """Generate word lists, explanations, and gap-filling exercises from knowledge base."""

    def __init__(self, chunks: List[Chunk]):
        """Initialize with knowledge base chunks."""
        self.chunks = chunks

    def extract_key_terms(self, limit: int = 20) -> List[Dict[str, str]]:
        """
        Extract key English terms and phrases from the knowledge base.
        Returns a list of dicts with 'term', 'definition', and 'source' keys.
        """
        terms = {}
        
        for chunk in self.chunks:
            # Extract terms from headings
            heading = chunk.heading or ""
            if heading:
                heading_terms = self._parse_heading_for_terms(heading)
                for term in heading_terms:
                    if term not in terms:
                        terms[term] = {
                            "term": term,
                            "definition": f"(from {chunk.source}: {heading})",
                            "source": chunk.source,
                            "heading": heading
                        }
            
            # Extract terms from text (simple heuristic: capitalized phrases)
            text_terms = self._extract_capitalized_phrases(chunk.text)
            for term in text_terms[:3]:  # Limit per chunk
                if term not in terms:
                    terms[term] = {
                        "term": term,
                        "definition": f"(from {chunk.source})",
                        "source": chunk.source,
                        "heading": heading
                    }
        
        result = list(terms.values())[:limit]
        return result

    def generate_word_list_with_translations(self, topic: str = None) -> str:
        """
        Generate a formatted word list with explanations for a given topic.
        If topic is None, generates from all chunks.
        """
        relevant_chunks = self.chunks
        if topic:
            relevant_chunks = [
                c for c in self.chunks
                if topic.lower() in c.heading.lower() or topic.lower() in c.text.lower()
            ]
        
        if not relevant_chunks:
            return "No relevant vocabulary found for the given topic."
        
        terms = self.extract_key_terms(limit=10)
        
        output = f"**Vocabulary List**\n"
        if topic:
            output += f"Topic: {topic}\n"
        output += "=" * 50 + "\n\n"
        
        for i, term_info in enumerate(terms, 1):
            output += f"{i}. **{term_info['term']}**\n"
            output += f"   Definition: {term_info['definition']}\n"
            output += f"   Source: {term_info['source']}\n\n"
        
        return output

    def create_fill_the_gaps_exercise(self, target_length: int = 3) -> Dict[str, any]:
        """
        Create a fill-the-gaps exercise from knowledge base content.
        Returns a dict with 'original_text', 'gapped_text', 'answers', and 'level'.
        """
        if not self.chunks:
            return {"error": "No content available for creating exercises."}
        
        # Find a reasonably long chunk
        suitable_chunks = [c for c in self.chunks if len(c.text.split()) > 30]
        if not suitable_chunks:
            suitable_chunks = self.chunks
        
        chunk = suitable_chunks[0]
        sentences = re.split(r'[.!?]+', chunk.text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 5]
        
        if not sentences:
            return {"error": "Could not generate exercise from available content."}
        
        # Take first few sentences
        original_text = ". ".join(sentences[:target_length]) + "."
        
        # Extract words to remove (target_length words)
        words = re.findall(r'\b\w+\b', original_text)
        if len(words) < target_length:
            target_length = len(words) // 2
        
        # Select words to remove (every nth word, avoiding small words)
        words_to_remove = []
        gap_indices = []
        step = max(1, len(words) // target_length)
        
        for i in range(0, len(words), step):
            if len(words_to_remove) < target_length and len(words[i]) > 3:
                words_to_remove.append(words[i])
                gap_indices.append(i)
        
        # Create gapped text
        gapped_text = original_text
        for word in words_to_remove:
            gapped_text = gapped_text.replace(word, "______", 1)
        
        return {
            "original_text": original_text,
            "gapped_text": gapped_text,
            "answers": words_to_remove,
            "level": "intermediate",
            "source": chunk.source,
            "heading": chunk.heading
        }

    def _parse_heading_for_terms(self, heading: str) -> List[str]:
        """Extract potential terms from a heading."""
        # Remove common words and split
        common_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'for', 'with'}
        words = heading.lower().split()
        terms = [w.strip('()[]{}') for w in words if w not in common_words and len(w) > 3]
        return terms

    def _extract_capitalized_phrases(self, text: str) -> List[str]:
        """Extract capitalized phrases from text (likely proper nouns or important terms)."""
        # Match capitalized words or phrases
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return matches[:5]


def format_word_list_for_chat(terms: List[Dict[str, str]]) -> str:
    """Format extracted word list for chat display."""
    output = "📚 **Word List from Knowledge Base:**\n\n"
    for i, term_info in enumerate(terms, 1):
        output += f"{i}. **{term_info['term']}**\n"
        output += f"   → {term_info['definition']}\n"
    return output


def format_gap_exercise_for_chat(exercise: Dict) -> str:
    """Format a fill-the-gaps exercise for chat display."""
    if "error" in exercise:
        return f"❌ {exercise['error']}"
    
    output = "✏️ **Fill the Gaps Exercise**\n\n"
    output += f"Text (from {exercise['source']}):\n"
    output += f"```\n{exercise['gapped_text']}\n```\n\n"
    output += f"Difficulty: {exercise['level']}\n"
    output += f"Find the words that replace the blanks above!\n"
    return output
