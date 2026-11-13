"""The short examples of chapter 5, in the order of the text.

Usage: uv run python -m guide_nlp.chapter05.snippets
"""

import re

STRONG = re.compile(r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{12,}$")
DOUBLED = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
QUOTED = re.compile(r"""(['"])(.*?)\1""")


def stacked_lookaheads() -> None:
    """Several conditions on the same text, each a lookahead."""
    print(bool(STRONG.match("motdepasse2024")))  # False: no uppercase
    print(bool(STRONG.match("MotDePasse2024")))  # True


def backreferences() -> None:
    """A group reused later in the same pattern."""
    print(DOUBLED.sub(r"\1", "Il a a vu le le chat"))  # Il a vu le chat
    print(QUOTED.findall("""Il dit "oui" puis 'non'."""))
    # [('"', 'oui'), ("'", 'non')]


def main() -> None:
    for demo in (stacked_lookaheads, backreferences):
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
