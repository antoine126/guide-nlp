"""The first listing of the book (foreword).

Usage: uv run python -m guide_nlp.chapter01.hello
"""


def greet(name: str) -> str:
    """Return a polite greeting."""
    return f"Bonjour, {name} !"


if __name__ == "__main__":
    print(greet("lectrice, lecteur"))
