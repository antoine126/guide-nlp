"""Reading an invoice image with a small vision-language model (chapter 44).

Usage: uv run --group llm python -m guide_nlp.chapter44.vlm_extract
(with `ollama pull qwen2.5vl:3b`)
"""

import base64
import io

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from guide_nlp.config import settings

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


class Facture(BaseModel):
    numero: str | None
    date: str | None
    total: float | None
    fournisseur: str | None
    ville: str | None


def extract(image: Image.Image, prompt: str, model: str = "qwen2.5vl:3b") -> Facture:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    reply = httpx.post(
        f"{settings.ollama_url}/api/chat",
        timeout=600,
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "format": Facture.model_json_schema(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(buffer.getvalue()).decode()],
                }
            ],
        },
    )
    reply.raise_for_status()
    return Facture.model_validate_json(reply.json()["message"]["content"])


def render(text: str, size: int = 26, width: int = 1000) -> Image.Image:
    """Draw a text document on a white page."""
    try:
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont = ImageFont.truetype(FONT, size)
    except OSError:
        font = ImageFont.load_default(size)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            if font.getlength(f"{current} {word}") > width - 80:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        lines.append(current)
    image = Image.new("RGB", (width, max(int(size * 1.5 * len(lines) + 120), width // 2)), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((40, 50 + int(i * size * 1.5)), line, fill=(20, 20, 20), font=font)
    return image


def main() -> None:
    text = (
        "Bonjour Sophie Girard,\nVeuillez trouver ci-joint la facture FA-2021-2208 de "
        "Plomberie Marchand datée du 26 octobre 2021. Un acompte de 2 665,75 euros a déjà "
        "été versé sur un total de 8 885,84 euros.\nCordialement,\nLe service comptable, Rennes."
    )
    prompt = (
        "Extrais les champs de cette facture : numero, date (AAAA-MM-JJ), total (montant "
        "TTC, nombre), fournisseur, ville. Mets null si l'information est absente."
    )
    print(extract(render(text), prompt))


if __name__ == "__main__":
    main()
