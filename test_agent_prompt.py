"""
Offline checks for Mykola persona and typo-handling prompt rules.
Run:  python test_agent_prompt.py
"""

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

    print("all prompt checks passed")


if __name__ == "__main__":
    main()
