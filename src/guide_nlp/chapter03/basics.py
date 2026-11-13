"""Characters, bytes and codecs (chapter 3).

Usage: uv run python -m guide_nlp.chapter03.basics
"""


def main() -> None:
    word = "café"  # 'café'
    data = word.encode("utf-8")
    print(len(word), len(data))  # 4 5
    print(data)  # b'caf\xc3\xa9'
    print(data.decode("utf-8"))  # café

    # Decoding with the wrong codec does not fail: it silently corrupts.
    print(data.decode("cp1252"))  # cafÃ©

    # Decoding invalid UTF-8 fails loudly, unless told otherwise.
    latin = word.encode("latin-1")  # b'caf\xe9'
    print(latin.decode("utf-8", errors="replace"))  # caf�


if __name__ == "__main__":
    main()
