"""Image search by similarity with DINOv2 and FAISS (chapter 42).

Usage: uv run --group neural python -m guide_nlp.chapter42.image_search
"""

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from guide_nlp.chapter42.images import open_rgb, validation_images
from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

name = "facebook/dinov2-base"
device = resolve_device(settings.device)
model = AutoModel.from_pretrained(name).to(device).eval()
processor = AutoImageProcessor.from_pretrained(name)


@torch.inference_mode()
def encode(images: list[Image.Image]) -> np.ndarray:
    batch = processor(images=images, return_tensors="pt").to(device)
    cls = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()


def main(per_class: int = 30) -> None:
    items = validation_images(per_class)
    gallery = [open_rgb(p) for p, _ in items]
    vectors = np.concatenate([encode(gallery[i : i + 64]) for i in range(0, len(gallery), 64)])
    index = faiss.IndexFlatIP(768)
    index.add(vectors)
    hits = 0
    for q in range(len(items)):
        _, ids = index.search(vectors[q : q + 1], 6)
        neighbours = [i for i in ids[0] if i != q][:5]
        hits += sum(items[i][1] == items[q][1] for i in neighbours)
    print(
        f"{len(items)} images: {hits / (5 * len(items)):.1%} of the 5 nearest neighbours "
        "share the class of the query"
    )


if __name__ == "__main__":
    main()
