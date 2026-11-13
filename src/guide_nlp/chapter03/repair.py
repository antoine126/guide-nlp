"""Read a file of unknown encoding and repair mojibake (chapter 3).

Usage: uv run python -m guide_nlp.chapter03.repair [FILE]
"""

import sys
from pathlib import Path

import ftfy
from charset_normalizer import from_bytes


def read_text_any(path: Path) -> str:
    """Read a text file whose encoding is unknown, then repair it."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")  # fast path: most files
    except UnicodeDecodeError:
        best = from_bytes(raw).best()
        if best is None:
            raise ValueError(f"{path} does not look like text") from None
        text = str(best)
    return ftfy.fix_text(text)


def main() -> None:
    broken = "lâ€™Ã©tÃ© dernier"  # mojibake of "l’été dernier"
    print(ftfy.fix_text(broken))  # l'été dernier
    if len(sys.argv) > 1:
        print(read_text_any(Path(sys.argv[1]))[:500])


if __name__ == "__main__":
    main()
