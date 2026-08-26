"""
Live-input sanity checks for the grading console. Kept separate from app.py
(rather than inline in the Streamlit page) so it can be imported by tests
without triggering Streamlit page-execution (st.set_page_config etc.) on import.

Note: rcaj-x-hardening-plan/04 suggests reusing preprocessing.py's `glossary` for
a gibberish check — that glossary is built only from rubric-criterion nouns
(a handful of subject-specific terms), not a general dictionary, so it can't
tell "genuine but off-topic language" from "gibberish". The heuristic below is
a crude, dependency-free proxy instead: alphabetic-character ratio, average
word length, and repeated-character runs.
"""


def crude_gibberish_score(text: str) -> float:
    """Fraction of characters that are alphabetic."""
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / len(text)


def input_sanity_check(answer_text: str) -> list[str]:
    issues = []
    stripped = answer_text.strip()
    words = stripped.split()
    word_count = len(words)

    if word_count < 3:
        issues.append("Answer is extremely short (<3 words) — score reliability is low.")
    if word_count > 500:
        issues.append("Answer is unusually long (>500 words) — outside typical training range.")

    if stripped:
        alpha_ratio = crude_gibberish_score(stripped)
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        has_long_repeat = any(len(set(w.lower())) <= 2 and len(w) >= 5 for w in words)
        if alpha_ratio < 0.6 or avg_word_len > 15 or has_long_repeat:
            issues.append(
                "Answer text looks unusual (low alphabetic ratio, abnormal word lengths, "
                "or repeated characters) — may not be genuine language."
            )

    return issues
