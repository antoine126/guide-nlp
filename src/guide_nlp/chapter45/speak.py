"""Speech synthesis with Piper, a small local voice (chapter 45).

Usage: uv run --group audio python -m guide_nlp.chapter45.speak ["Texte à dire."]
Writes data/interim/voice/bonjour.wav.
"""

import sys
import wave
from pathlib import Path

from huggingface_hub import hf_hub_download
from piper import PiperVoice

from guide_nlp.config import settings

VOICE = settings.interim_dir / "voice"


def say(text: str, output: Path) -> Path:
    path = "fr/fr_FR/siwis/medium/fr_FR-siwis-medium"
    onnx = hf_hub_download("rhasspy/piper-voices", path + ".onnx")
    hf_hub_download("rhasspy/piper-voices", path + ".onnx.json")
    voice = PiperVoice.load(onnx)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        voice.synthesize_wav(text, wav)
    return output


if __name__ == "__main__":
    print(say(sys.argv[1] if len(sys.argv) > 1 else "Bonjour.", VOICE / "bonjour.wav"))
