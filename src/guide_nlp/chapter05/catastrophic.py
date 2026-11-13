"""Catastrophic backtracking, and two ways out (chapter 5).

Usage: uv run python -m guide_nlp.chapter05.catastrophic
The evil pattern doubles its time with each extra 'a': n=26 takes seconds.
"""

import re
import time

EVIL = re.compile(r"^(a+)+$")
ATOMIC = re.compile(r"^(?>a+)+$")  # Python >= 3.11
SAFE = re.compile(r"^a+$")


def timed(pattern: re.Pattern[str], text: str) -> float:
    start = time.perf_counter()
    pattern.match(text)
    return time.perf_counter() - start


def main() -> None:
    for n in range(16, 27, 2):
        text = "a" * n + "!"
        print(
            f"n={n:<3} evil={timed(EVIL, text):9.4f}s "
            f"atomic={timed(ATOMIC, text):.6f}s safe={timed(SAFE, text):.6f}s"
        )


if __name__ == "__main__":
    main()
