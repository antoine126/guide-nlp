"""Zero-shot image classification with SigLIP (chapter 43).

Usage: uv run --group neural python -m guide_nlp.chapter43.zero_shot
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from guide_nlp.chapter42.images import CLASSES, open_rgb, validation_images
from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

name = "google/siglip-base-patch16-224"
device = resolve_device(settings.device)
model = AutoModel.from_pretrained(name).to(device).eval()
processor = AutoProcessor.from_pretrained(name)
labels = ["une église", "un parachute", "une balle de golf"]


@torch.inference_mode()
def classify(image: Image.Image, labels: list[str] = labels) -> str:
    # SigLIP expects max_length padding, as it was trained
    texts = processor(
        text=[f"une photo de {x}." for x in labels],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    pixels = processor(images=image, return_tensors="pt").to(device)
    # Transformers 5 returns an output object here, not a tensor
    text_vectors = model.get_text_features(**texts).pooler_output
    image_vectors = model.get_image_features(**pixels).pooler_output
    scores = torch.nn.functional.cosine_similarity(image_vectors, text_vectors)
    return labels[int(scores.argmax())]


def main(per_class: int = 20) -> None:
    all_labels = list(CLASSES.values())
    items = validation_images(per_class)
    right = sum(classify(open_rgb(p), all_labels) == label for p, label in items)
    print(f"zero-shot accuracy over the 10 classes: {right / len(items):.1%} ({len(items)} images)")


if __name__ == "__main__":
    main()
