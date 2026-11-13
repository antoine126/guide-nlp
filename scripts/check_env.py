"""Check that the environment is ready (chapter 2).

Usage: uv run --all-groups python scripts/check_env.py
"""

from importlib.metadata import PackageNotFoundError, version

from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

PACKAGES = ["regex", "spacy", "scikit-learn", "torch", "transformers"]


def main() -> None:
    for name in PACKAGES:
        try:
            print(f"{name:<14} {version(name)}")
        except PackageNotFoundError:
            print(f"{name:<14} not installed (see dependency groups)")
    print(f"{'device':<14} {resolve_device(settings.device)}")

    import spacy  # imported late: optional dependency

    try:
        spacy.load(settings.spacy_model)
        print(f"{'spacy model':<14} {settings.spacy_model} OK")
    except OSError:
        print(f"{'spacy model':<14} {settings.spacy_model} missing")


if __name__ == "__main__":
    main()
