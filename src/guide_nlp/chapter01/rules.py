"""Era 1: hand-written rules (chapter 1).

Usage: uv run python -m guide_nlp.chapter01.rules
The third message is a false positive: "ne regrette plus" is not a complaint.
"""

import re

from guide_nlp.chapter01.messages import MESSAGES

# Each pattern is a hint that the customer is unhappy.
COMPLAINT_PATTERNS = [
    r"\bne\s+[\w']+\s+plus\b",  # "ne s'allume plus"
    r"\bpas\s+terrible\b",
    r"\b(?:rembours|panne|cass)\w*",  # refund, breakdown, broken
]
COMPLAINT_RE = re.compile("|".join(COMPLAINT_PATTERNS), re.IGNORECASE)


def is_complaint(message: str) -> bool:
    """Flag a message as a complaint with hand-written rules."""
    return COMPLAINT_RE.search(message) is not None


def main() -> None:
    for message in MESSAGES:
        print(is_complaint(message), "|", message)


if __name__ == "__main__":
    main()
