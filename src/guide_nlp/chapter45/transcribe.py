"""Transcription with faster-whisper, voice activity detection on (chapter 45).

Usage: uv run --group audio python -m guide_nlp.chapter45.transcribe [AUDIO]
"""

import sys
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

from guide_nlp.chapter45.speak import VOICE, say


def transcribe(path: Path, model_name: str = "large-v3-turbo") -> list[tuple[float, float, str]]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=12)
    # faster-whisper 1.2 decodes files with an option that PyAV 19 no longer has:
    # decode the audio ourselves (16 kHz mono) and pass the array.
    audio, _ = librosa.load(path, sr=16000, mono=True)
    segments, _ = model.transcribe(audio, language="fr", vad_filter=True, word_timestamps=True)
    return [(s.start, s.end, s.text.strip()) for s in segments]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio = Path(sys.argv[1])
    else:
        audio = say(
            "Bonjour à tous. Nous repoussons le lancement au 15 mars.", VOICE / "reunion.wav"
        )
    for start, end, text in transcribe(audio):
        print(f"[{start:.1f} - {end:.1f}] {text}")
