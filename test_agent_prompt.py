"""
Offline checks for Mykola persona and typo-handling prompt rules.
Run:  python test_agent_prompt.py
"""

import datetime

from agent import SYSTEM_PROMPT, _personalized_system


def main() -> None:
    prompt = SYSTEM_PROMPT.lower()

    # Issue #31: no "buddy" wording in persona text.
    assert "buddy" not in prompt, "SYSTEM_PROMPT must not use 'buddy'"

    # Persona should frame Mykola as companion/guide.
    assert "english companion" in prompt, "Expected 'English companion' wording"
    assert "study guide" in prompt, "Expected 'study guide' wording"

    # Silent typo handling rules must be explicit.
    assert "silently interpret" in prompt, "Missing silent typo interpretation rule"
    assert "do not point out" in prompt, "Missing no-typo-comment rule"
    assert "intended meaning" in prompt, "Missing intended-meaning response rule"

    # Personalization should preserve the same base rules.
    personalized = _personalized_system("Olena").lower()
    assert "buddy" not in personalized, "Personalized prompt must not use 'buddy'"
    assert "silently interpret" in personalized, "Personalized prompt lost typo rules"

    # Issue #35: persona traits must be spelled out, not implied.
    for trait in (
        "born december 13, 1981",
        "a gentleman of intellect, courtesy, and art",
        "leontovych",            # heritage
        "british",               # poise
        "french",                # second language
        "choir conducting",      # musical credentials
        "the beatles",           # musical illustrations...
        "queen",
        "depeche mode",
        "faux ami",              # French connections in practice
        "never reproduce full lyrics",  # copyright guardrail
        "one flourish per reply",       # moderation of the persona
        "mnemonic",              # original verse trait
    ):
        assert trait in prompt, f"SYSTEM_PROMPT missing persona trait: {trait!r}"

    # Voice examples exist but must not be parroted.
    assert "never repeat these verbatim" in prompt, "Missing verbatim guard on examples"

    # Issue #48: Mykola must know he is Claude-powered and own up to being an AI.
    assert "claude" in prompt, "SYSTEM_PROMPT must mention Claude"
    assert "anthropic" in prompt, "SYSTEM_PROMPT must mention Anthropic"
    assert "powered by anthropic's claude" in prompt, \
        "Missing the Claude-powered identity line"
    assert "never claim to be human" in prompt, "Missing the never-claim-human rule"
    assert "never deny being an ai" in prompt, "Missing the never-deny-AI rule"

    # Issue #40: symbolic birthday and age handling should be explicit.
    dynamic = _personalized_system().lower()
    assert "age handling:" in dynamic, "Missing explicit age handling section"
    assert "symbolic birthday: december 13, 1981." in dynamic, "Missing symbolic birthday guidance"

    today = datetime.date.today()
    expected_age = today.year - 1981 - ((today.month, today.day) < (12, 13))
    assert f"symbolic age is {expected_age}." in dynamic, "Missing current symbolic age value"
    assert "recalculate the age whenever the current date changes." in dynamic, "Missing age recalculation rule"

    print("all prompt checks passed")


def check_knowledge_base() -> None:
    """The persona must be backed by retrievable knowledge, not just prompt text."""
    from rag import KnowledgeBase

    kb = KnowledgeBase()
    sources = {c.source for c in kb.chunks}
    for doc in ("french_connections.md", "british_english.md", "music_and_english.md"):
        assert doc in sources, f"Knowledge base missing {doc}"

    # On-trait questions should retrieve the matching document.
    cases = {
        "Why does English have so many French words?": "french_connections.md",
        "What is the difference between British and American spelling of colour?": "british_english.md",
        "How can I learn English with songs?": "music_and_english.md",
        "Who was Mykola Leontovych and what is Shchedryk?": "music_and_english.md",
    }
    for question, expected in cases.items():
        top = kb.retrieve(question, top_k=3)
        got = [c.source for c in top]
        assert expected in got, f"{question!r} retrieved {got}, expected {expected}"

    print("all knowledge-base checks passed")


if __name__ == "__main__":
    main()
    check_knowledge_base()
