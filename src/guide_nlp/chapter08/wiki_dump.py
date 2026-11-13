"""Stream the articles of a MediaWiki XML dump in constant memory (chapter 8).

Usage: uv run python -m guide_nlp.chapter08.wiki_dump DUMP.xml
(e.g. frwiki-latest-pages-articles1.xml, decompressed)
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def iter_articles(path: Path) -> Iterator[tuple[str, str]]:
    """Stream (title, wikitext) pairs of main-namespace pages."""
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == f"{NS}page":
            if elem.findtext(f"{NS}ns") == "0":  # articles only
                title = elem.findtext(f"{NS}title") or ""
                text = elem.findtext(f"{NS}revision/{NS}text") or ""
                yield title, text
            elem.clear()  # free memory


if __name__ == "__main__":
    for n, (title, text) in enumerate(iter_articles(Path(sys.argv[1]))):
        print(f"{title[:50]:<50} {len(text):>8} characters")
        if n == 9:
            break
