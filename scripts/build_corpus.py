"""Turn the raw Wikipedia collection into the clean 'Articles' corpus (chapter 10).

Usage:
    uv run --group web python -m guide_nlp.corpus.collect
        # -> data/raw/wikipedia/articles_raw.jsonl
    uv run --group web python scripts/build_corpus.py [RAW.jsonl]

Prints the filtering funnel, then writes data/processed/articles.jsonl,
articles.parquet and stats.json. The book's corpus (collected on 2026-11-02)
is already in data/processed/: rebuilding it from a new collection gives a
slightly different corpus, since Wikipedia changes every day.
"""

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHashLSH
from lingua import Language, LanguageDetectorBuilder

from guide_nlp.chapter04.entities import EMAIL, PHONE_FR
from guide_nlp.chapter10.minhash import jaccard, minhash_of
from guide_nlp.chapter10.snippets import fingerprint
from guide_nlp.config import settings
from guide_nlp.corpus.clean import strip_tail
from guide_nlp.corpus.quality import WORD, quality_issue
from guide_nlp.text.normalize import normalize_text

THRESHOLD = 0.5


def shingles(text: str, k: int = 3) -> set[str]:
    """Word 3-grams; words keep their apostrophes and hyphens (quality.WORD).

    This is the tokenisation behind the book's numbers (25 LSH candidates,
    2,923 articles); chapter10.minhash.shingles, which splits "l'été" in two,
    finds 20 candidates and keeps 2,925 articles.
    """
    words = [w.lower() for w in WORD.findall(text)]
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def clean_text(text: str) -> str:
    text = normalize_text(text)
    return strip_tail(re.sub(r"\n{3,}", "\n\n", text))


def main(raw_path: Path, out_dir: Path) -> None:
    articles = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    funnel = [("Pages collectées", len(articles))]
    print(funnel[-1])

    # 1. Boilerplate: reference sections at the end of the page.
    before = sum(len(a["text"]) for a in articles)
    for a in articles:
        a["text"] = clean_text(a["text"])
    after = sum(len(a["text"]) for a in articles)
    print(f"characters removed: {1 - after / before:.1%}")

    # 2. Language.
    detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ENGLISH, Language.GERMAN, Language.SPANISH, Language.ITALIAN
    ).build()
    languages: Counter[str] = Counter()
    kept = []
    for a in articles:
        language = detector.detect_language_of(a["text"][:2000]) if a["text"] else None
        name = language.name if language else "NONE"
        languages[name] += 1
        if name == "FRENCH":
            kept.append(a)
    articles = kept
    funnel.append(("Langue française", len(articles)))
    print(funnel[-1], dict(languages))

    # 3. Quality.
    issues: Counter[str] = Counter()
    kept = []
    for a in articles:
        issue = quality_issue(a["text"]) if a["text"] else "trop court"
        if issue:
            issues[issue] += 1
        else:
            kept.append(a)
    articles = kept
    funnel.append(("Filtres de qualité", len(articles)))
    print(funnel[-1], dict(issues))

    # 4. Exact duplicates.
    seen: set[str] = set()
    kept = []
    for a in articles:
        key = fingerprint(a["text"])
        if key not in seen:
            seen.add(key)
            kept.append(a)
    articles = kept
    funnel.append(("Doublons exacts", len(articles)))
    print(funnel[-1])

    # 5. Near duplicates: MinHash + LSH, verified with the exact Jaccard.
    articles.sort(key=lambda a: -len(a["text"]))  # keep the longest of a group
    sets = [shingles(a["text"]) for a in articles]
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
    kept_idx, candidates, examples = [], 0, []
    for i, items in enumerate(sets):
        if not items:
            continue
        m = minhash_of(items)
        twins = cast(list[int], lsh.query(m))
        candidates += len(twins)
        real = [j for j in twins if jaccard(items, sets[j]) >= THRESHOLD]
        if real:
            j = real[0]
            examples.append(
                (articles[i]["title"], articles[j]["title"], round(jaccard(items, sets[j]), 3))
            )
        else:
            lsh.insert(i, m)
            kept_idx.append(i)
    print(f"LSH candidate pairs: {candidates}, confirmed: {len(examples)}")
    for example in examples:
        print("  near-duplicate:", example)
    articles = [articles[i] for i in kept_idx]
    funnel.append(("Doublons approximatifs", len(articles)))
    print(funnel[-1])

    # 6. Personal data: mask e-mail addresses and phone numbers.
    n_mail = n_phone = 0
    for a in articles:
        a["text"], x = EMAIL.subn("<EMAIL>", a["text"])
        a["text"], y = PHONE_FR.subn("<TEL>", a["text"])
        n_mail, n_phone = n_mail + x, n_phone + y
    print(f"masked: {n_mail} e-mails, {n_phone} phone numbers")

    # 7. Stratified split, after deduplication.
    rng = random.Random(42)
    by_rubric: defaultdict[str, list[dict]] = defaultdict(list)
    for a in sorted(articles, key=lambda a: a["pageid"]):
        by_rubric[a["rubric"]].append(a)
    for items in by_rubric.values():
        rng.shuffle(items)
        n_test = round(len(items) * 0.2)
        for k, a in enumerate(items):
            a["split"] = "test" if k < n_test else "train"
    articles = [a for items in by_rubric.values() for a in items]

    # 8. Store.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "articles.jsonl").open("w", encoding="utf-8") as sink:
        for a in articles:
            sink.write(json.dumps(a, ensure_ascii=False) + "\n")
    pq.write_table(pa.Table.from_pylist(articles), out_dir / "articles.parquet", compression="zstd")
    n_words = [len(WORD.findall(a["text"])) for a in articles]
    stats = {
        "funnel": funnel,
        "per_rubric": dict(Counter(a["rubric"] for a in articles)),
        "per_split": dict(Counter(a["split"] for a in articles)),
        "words_total": sum(n_words),
        "words_median": statistics.median(n_words),
        "jsonl_MB": round((out_dir / "articles.jsonl").stat().st_size / 1e6, 1),
        "parquet_MB": round((out_dir / "articles.parquet").stat().st_size / 1e6, 1),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    default = settings.raw_dir / "wikipedia" / "articles_raw.jsonl"
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else default,
        Path(sys.argv[2]) if len(sys.argv) > 2 else settings.processed_dir,
    )
