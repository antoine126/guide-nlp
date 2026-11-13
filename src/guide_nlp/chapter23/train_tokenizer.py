"""Train a BPE tokenizer on the corpus with Hugging Face tokenizers (chapter 23).

Usage: uv run --group neural python -m guide_nlp.chapter23.train_tokenizer
"""

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def corpus_file() -> Path:
    """One training article per line, written once."""
    path = settings.processed_dir / "articles.txt"
    if not path.exists():
        lines = (a.text.replace("\n", " ") for a in load_articles("train"))
        path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=8000, special_tokens=SPECIALS, end_of_word_suffix="</w>"
    )
    tokenizer.train([str(corpus_file())], trainer)
    print(tokenizer.encode("anticonstitutionnellement").tokens)


if __name__ == "__main__":
    main()
