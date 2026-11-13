"""The short examples of chapter 4, in the order of the text.

Usage: uv run python -m guide_nlp.chapter04.snippets
"""

import re


def raw_strings() -> None:
    """Always write patterns as raw strings."""
    print(re.search("\bchat\b", "le chat dort"))  # None: "\b" is a backspace
    print(re.search(r"\bchat\b", "le chat dort"))  # <re.Match ... match='chat'>


def word_class() -> None:
    r"""``\w`` is Unicode-aware, unless ``re.ASCII`` is set."""
    print(re.findall(r"\w+", "Élise a 3 chats, l'aînée 12."))
    # ['Élise', 'a', '3', 'chats', 'l', 'aînée', '12']
    print(re.findall(r"\w+", "Élise a 3 chats", flags=re.ASCII))
    # ['lise', 'a', '3', 'chats']


def greedy_and_lazy() -> None:
    """Greedy, lazy and negated-class quantifiers."""
    html = "<b>gras</b> et <i>italique</i>"
    print(re.findall(r"<.*>", html))  # ['<b>gras</b> et <i>italique</i>']
    print(re.findall(r"<.*?>", html))  # ['<b>', '</b>', '<i>', '</i>']
    print(re.findall(r"<[^>]*>", html))  # ['<b>', '</b>', '<i>', '</i>']


def anchors() -> None:
    """Line anchors and word boundaries."""
    log = """INFO start
ERREUR disque plein
INFO retry
ERREUR timeout"""
    print(re.findall(r"^ERREUR (.+)$", log))  # []
    print(re.findall(r"^ERREUR (.+)$", log, flags=re.M))  # ['disque plein', 'timeout']
    print(re.findall(r"\bchat\b", "chat, chaton, achat, le chat"))  # ['chat', 'chat']


def alternation() -> None:
    """The alternation has the lowest precedence: group it."""
    for word in ("chat", "chaton", "chien", "un chien"):
        loose = bool(re.search(r"^chat|chien$", word))
        strict = bool(re.search(r"^(?:chat|chien)$", word))
        print(f"{word:<9} {loose!s:<6} {strict}")
    # chat      True   True
    # chaton    True   False
    # chien     True   True
    # un chien  True   False


def mask_digits(match: re.Match[str]) -> str:
    """Keep the last two digits of a number, mask the others."""
    digits = match.group()
    return "*" * (len(digits) - 2) + digits[-2:]


def replacement_function() -> None:
    """``re.sub`` accepts a function instead of a template."""
    print(re.sub(r"\d{6,}", mask_digits, "Carte 4970123456781234, client 551203"))
    # Carte **************34, client ****03


def main() -> None:
    demos = (raw_strings, word_class, greedy_and_lazy, anchors, alternation, replacement_function)
    for demo in demos:
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
