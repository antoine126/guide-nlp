"""PDF to Markdown with docling: headings, lists, tables (chapter 8).

docling pins an older huggingface-hub: it has its own environment.
Usage:
    uv sync --project envs/docling
    uv run --project envs/docling python -m guide_nlp.chapter08.docling_convert [PDF]
"""

import sys

from docling.document_converter import DocumentConverter


def main(path: str) -> None:
    result = DocumentConverter().convert(path)
    markdown = result.document.export_to_markdown()  # headings, lists, tables
    print(markdown[:2000])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/samples/rapport.pdf")
