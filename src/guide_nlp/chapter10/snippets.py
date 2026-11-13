"""The short examples of chapter 10, in the order of the text.

Usage: uv run --group neural python -m guide_nlp.chapter10.snippets
"""

import hashlib
import re

from lingua import Language, LanguageDetectorBuilder

from guide_nlp.config import settings


def detect_language() -> None:
    """Restrict the detector to the languages you expect: faster and more accurate."""
    detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ENGLISH, Language.GERMAN, Language.SPANISH, Language.ITALIAN
    ).build()
    print(detector.detect_language_of("Le calendrier mésoaméricain est un système"))


def fingerprint(text: str) -> str:
    """Exact duplicates up to case, spaces and punctuation."""
    return hashlib.sha1(re.sub(r"\W+", "", text.casefold()).encode()).hexdigest()


def exact_duplicates() -> None:
    print(fingerprint("Le chat dort.") == fingerprint("le  chat, dort"))  # True


def parquet_with_datasets() -> None:
    """Load the corpus as a Hugging Face Dataset and filter it in parallel."""
    from datasets import load_dataset

    path = str(settings.processed_dir / "articles.parquet")
    corpus = load_dataset("parquet", data_files=path, split="train")
    short = corpus.filter(lambda a: len(a["text"]) < 2000, num_proc=4)
    print(len(corpus), len(short))


def main() -> None:
    for demo in (detect_language, exact_duplicates, parquet_with_datasets):
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
