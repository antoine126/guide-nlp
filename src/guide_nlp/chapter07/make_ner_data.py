"""Annotated spans to spaCy's binary training format (chapter 7).

Usage: uv run --group neural python -m guide_nlp.chapter07.make_ner_data
Writes data/processed/train.spacy and dev.spacy from a sample of WikiNER, then:

    uv run python -m spacy init config config.cfg --lang fr --pipeline ner --optimize accuracy
    uv run python -m spacy train config.cfg --output models/ner \\
        --paths.train data/processed/train.spacy --paths.dev data/processed/dev.spacy
"""

import spacy
from spacy.tokens import DocBin

from guide_nlp.config import settings

# (text, [(start_char, end_char, label), ...]) -- e.g. exported from an
# annotation tool such as Label Studio or Prodigy.
Annotated = list[tuple[str, list[tuple[int, int, str]]]]


def to_docbin(examples: Annotated, path: str) -> None:
    nlp = spacy.blank("fr")
    db = DocBin()
    for text, spans in examples:
        doc = nlp.make_doc(text)
        ents = [doc.char_span(s, e, label=lab, alignment_mode="contract") for s, e, lab in spans]
        doc.ents = [ent for ent in ents if ent is not None]
        db.add(doc)
    db.to_disk(path)


def from_io_tags(tokens: list[str], tags: list[str]) -> tuple[str, list[tuple[int, int, str]]]:
    """WikiNER uses IO tags: a run of identical non-O tags is one entity."""
    text, spans, position = "", [], 0
    current: list[int | str] | None = None
    for token, tag in zip(tokens, tags, strict=True):
        start = position
        text += token + " "
        position += len(token) + 1
        if current is not None and tag == current[2]:
            current[1] = start + len(token)
            continue
        if current is not None:
            spans.append((int(current[0]), int(current[1]), str(current[2])))
            current = None
        if tag != "O":
            current = [start, start + len(token), tag]
    if current is not None:
        spans.append((int(current[0]), int(current[1]), str(current[2])))
    return text.rstrip(), spans


def main(n_train: int = 20_000, n_dev: int = 2_000) -> None:
    from datasets import load_dataset

    data = load_dataset("Jean-Baptiste/wikiner_fr")
    names = data["train"].features["ner_tags"].feature.names
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    for split, n, name in (("train", n_train, "train"), ("test", n_dev, "dev")):
        rows = data[split].shuffle(seed=0).select(range(n))
        examples = [from_io_tags(r["tokens"], [names[t] for t in r["ner_tags"]]) for r in rows]
        to_docbin(examples, str(out / f"{name}.spacy"))
        print(f"{name}: {len(examples)} sentences -> {out / f'{name}.spacy'}")


if __name__ == "__main__":
    main()
