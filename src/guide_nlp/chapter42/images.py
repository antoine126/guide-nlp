"""Imagenette, the image collection of chapters 42 and 43.

Download https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz and
extract it into data/raw/imagenette/ (ten ImageNet classes, 3,925 validation images).
"""

from pathlib import Path

from PIL import Image

from guide_nlp.config import settings

ROOT = settings.raw_dir / "imagenette" / "imagenette2-320"
CLASSES = {  # WordNet id -> French label
    "n01440764": "une tanche",
    "n02102040": "un épagneul anglais",
    "n02979186": "un lecteur de cassettes",
    "n03000684": "une tronçonneuse",
    "n03028079": "une église",
    "n03394916": "un cor d'harmonie",
    "n03417042": "un camion poubelle",
    "n03425413": "une pompe à essence",
    "n03445777": "une balle de golf",
    "n03888257": "un parachute",
}


def validation_images(per_class: int | None = None) -> list[tuple[Path, str]]:
    """(path, French label) pairs, class by class."""
    out = []
    for wnid, label in CLASSES.items():
        paths = sorted((ROOT / "val" / wnid).glob("*.JPEG"))[:per_class]
        out += [(p, label) for p in paths]
    return out


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
