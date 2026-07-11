"""
Offline checks for the retrieval layer — no API key or network needed.
Run:  python test_rag.py
"""

from rag import KnowledgeBase


def main() -> None:
    kb = KnowledgeBase()
    assert len(kb.chunks) >= 10, f"expected 10+ chunks, got {len(kb.chunks)}"

    # Grammar question -> grammar_tips.md
    results = kb.retrieve("when should I use present perfect instead of past simple")
    assert results, "no results for a grammar question"
    assert results[0].source == "grammar_tips.md", results[0]
    assert "Present Perfect" in results[0].heading

    # Verb forms question -> irregular_verbs.md
    results = kb.retrieve("what is the past participle of the verb drink")
    assert results[0].source == "irregular_verbs.md", results[0]

    # App question -> kuantorflow_faq.md
    results = kb.retrieve("how do I take a quiz in the app")
    assert results[0].source == "kuantorflow_faq.md", results[0]

    # Method question -> learning_strategies.md
    results = kb.retrieve("how many new words should I learn per day")
    assert results[0].source == "learning_strategies.md", results[0]

    # Irrelevant question -> below min_score, empty result
    results = kb.retrieve("zzz qqq xxx")
    assert results == [], f"expected no results for gibberish, got {results}"

    print(f"all retrieval checks passed ({len(kb.chunks)} chunks indexed)")


if __name__ == "__main__":
    main()
