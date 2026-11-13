"""Searching document pages as images with ColQwen2 (chapter 43).

ColPali needs torch < 2.14: it has its own environment.
Usage:
    uv sync --project envs/vision
    uv run --project envs/vision python -m guide_nlp.chapter43.colqwen_search [PDF]
"""

import sys
from pathlib import Path

import pymupdf
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

name = "vidore/colqwen2-v1.0"


def pages_as_images(pdf: Path, dpi: int = 96) -> list[Image.Image]:
    doc = pymupdf.open(pdf)
    out = []
    for page in doc:
        pixmap = page.get_pixmap(dpi=dpi)
        out.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    return out


def main(pdf: Path, question: str) -> None:
    model = ColQwen2.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = ColQwen2Processor.from_pretrained(name)
    images = pages_as_images(pdf)
    batches = [images[i : i + 4] for i in range(0, len(images), 4)]

    with torch.inference_mode():
        pages = [
            model(**processor.process_images(batch).to(model.device)).float().cpu()
            for batch in batches
        ]
        query = model(**processor.process_queries([question]).to(model.device)).float().cpu()
    scores = processor.score_multi_vector(list(query), [p for batch in pages for p in batch])
    best = scores[0].topk(min(3, len(images)))
    print(
        [(int(i) + 1, round(float(s), 2)) for s, i in zip(best.values, best.indices, strict=True)]
    )


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/samples/rapport.pdf")
    main(pdf, "Quelle est la hausse des dépenses en un an ?")
