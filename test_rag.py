"""
Offline checks for the retrieval layer — no API key or network needed.
Run:  python test_rag.py
"""

import tempfile
from pathlib import Path

import rag
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




def heading_weighting() -> None:
    """A section can be found by its own name (#82).

    Reported from the deployed site: asked about *Fill the gap*, Mykola said
    the excerpts did not describe it -- while the section describing it sat in
    the index, four paragraphs long.

    The names are the problem. `fill`, `gap`, `spell`, `odd`, `real` are the
    least rare words in a document about word games, so TF-IDF gave the *name*
    of a thing almost no weight, and the section that was the thing lost to the
    sections merely mentioning it.

    Both directions are checked, because only one of them is the fix. Finding a
    section by name is what was broken; keeping the sections whose answers live
    in a **body** is what a heavier weight could plausibly break, and a fix that
    found activities by losing everything else would be worse than the bug.
    """
    # Shaped like the real failure: the words in each heading are common across
    # the document, and no section repeats its own title in its prose -- which
    # is how a well-written guide reads.
    guide = chr(10).join([
        "## Fill the gap", "",
        "A word is cut out of one of its own example sentences and you supply",
        "it. Tick the ones you knew.", "",
        "## Spell it", "",
        "The word is read aloud and you type what you heard, letter for",
        "letter.", "",
        "## Odd one out", "",
        "Four words from two topics, and you pick the intruder.", "",
        "## Choosing what to practise", "",
        "Tick the topics you want. Every game asks the same question and you",
        "can fill a round from one topic or all of them. If a gap appears",
        "because one topic is thin, the gap is filled from another, so you",
        "never see an odd short round. Spell out as many topics as you like.",
        "",
        "## How long a round is", "",
        "Ten cards by default. A round can be shorter when a topic cannot",
        "spell out enough usable words: a gap in the deck leaves a gap in the",
        "round, and an odd number of cards is normal. Nothing is filled in for",
        "you.", "",
        "## Printable worksheets", "",
        "A worksheet prints the text with a gap where each word was, so a",
        "learner can fill the gaps by hand. Spell the answers in the margin.",
        "The gaps are sized to the word, and an odd gap is left where an",
        "expression runs long.", "",
        "## The review popup", "",
        "Cards are shown before they are saved. Fill in any field that is",
        "blank, and spell the word exactly as you want it stored.",
    ])

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "guide.md"
        doc.write_text(guide, encoding="utf-8")
        kb = KnowledgeBase(extra_docs=[doc])

        def headings(question):
            return [c.heading for c in kb.retrieve(question, top_k=3)]

        # By name, and phrased the way somebody actually asks.
        for question, wanted in (
                ("Fill the gap", "Fill the gap"),
                ("tell me about the Fill the gap activity", "Fill the gap"),
                ("how does Fill the gap work", "Fill the gap"),
                ("what is Odd one out", "Odd one out"),
                ("explain Spell it", "Spell it"),
        ):
            assert wanted in headings(question), (
                f"{question!r} did not reach {wanted!r}: {headings(question)}")

        # The other direction: a question whose answer is in a body, under a
        # heading sharing none of its words.
        asked = "how many cards will I get"
        assert "How long a round is" in headings(asked), headings(asked)

    # A measured number, not a free parameter: 1 leaves three of ten real
    # activity questions unreachable, and anything above 3 buys nothing.
    assert rag.HEADING_WEIGHT == 3, rag.HEADING_WEIGHT

if __name__ == "__main__":
    main()
    heading_weighting()
