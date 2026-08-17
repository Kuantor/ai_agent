"""
Offline checks for the retrieval layer — no API key or network needed.
Run:  python test_rag.py
"""

import tempfile
from pathlib import Path

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

    # Method question -> learning_strategies.md
    results = kb.retrieve("how many new words should I learn per day")
    assert results[0].source == "learning_strategies.md", results[0]

    # Irrelevant question -> below min_score, empty result
    results = kb.retrieve("zzz qqq xxx")
    assert results == [], f"expected no results for gibberish, got {results}"

    check_injected_documents(kb)

    print(f"all retrieval checks passed ({len(kb.chunks)} chunks indexed)")


def check_injected_documents(plain: KnowledgeBase) -> None:
    """Documents the host hands in are indexed beside our own (kuantorflow#310).

    This is how Mykola knows the application he is embedded in: KuantorFlow
    passes its own user guide rather than keeping a copy of it here. There used
    to be such a copy — `kuantorflow_faq.md` — and it is gone because it had
    drifted into telling learners things that were no longer true. Nothing in
    this repo describes that app any more, deliberately: the app describes
    itself.

    Written to a temp directory rather than reading the real guide from a
    sibling checkout, so these checks still run with this repo on its own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "host-guide.md"
        doc.write_text(
            "# Host application\n\n"
            "## The wardrobe button\n\n"
            "Pressing the wardrobe button rearranges your waistcoats.\n",
            encoding="utf-8")

        # Without it, nothing here describes the host's app at all. Note what
        # this does *not* assert: that the question returns nothing. TF-IDF
        # answers "what does the wardrobe button do" from `french_connections`
        # at 0.175, comfortably over the 0.05 floor, because "wardrobe" is a
        # French borrowing. Retrieval always finds its nearest chunk — which is
        # exactly why an out-of-date document about the app is worse than none,
        # and why this one is injected rather than kept here.
        assert not any(c.source == "host-guide.md" for c in plain.chunks), \
            "the plain knowledge base should hold nothing of the host's app"

        kb = KnowledgeBase(extra_docs=[doc])
        results = kb.retrieve("what does the wardrobe button do")
        assert results, "an injected document should be retrievable"
        assert results[0].source == "host-guide.md", results[0]
        assert "waistcoats" in results[0].text, results[0]

        # Our own documents keep working alongside it.
        assert kb.retrieve(
            "what is the past participle of the verb drink"
        )[0].source == "irregular_verbs.md"

        # A path that is not there is a reason to know less, never a reason for
        # the widget to fail to load: the host may be an older checkout.
        missing = KnowledgeBase(extra_docs=[Path(tmp) / "nope.md", doc])
        assert missing.retrieve("what does the wardrobe button do"), \
            "a missing path must be skipped without losing the good ones"


if __name__ == "__main__":
    main()
