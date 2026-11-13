"""Reading robots.txt rules (chapter 9).

Usage: uv run python -m guide_nlp.chapter09.robots
"""

from urllib.robotparser import RobotFileParser

RULES = """User-agent: *
Disallow: /prive/
Crawl-delay: 1

User-agent: BadBot
Disallow: /
"""


def main() -> None:
    parser = RobotFileParser()
    parser.parse(RULES.splitlines())
    print(parser.can_fetch("guide-nlp-book", "https://exemple.fr/articles"))  # True
    print(parser.can_fetch("guide-nlp-book", "https://exemple.fr/prive/x"))  # False
    print(parser.can_fetch("BadBot", "https://exemple.fr/articles"))  # False
    print(parser.crawl_delay("guide-nlp-book"))  # 1


if __name__ == "__main__":
    main()
