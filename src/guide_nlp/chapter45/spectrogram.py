"""The log-mel spectrogram that speech models read (chapter 45).

Usage: uv run --group audio python -m guide_nlp.chapter45.spectrogram
"""

import librosa

from guide_nlp.chapter45.speak import VOICE, say


def main() -> None:
    path = VOICE / "bonjour.wav"
    if not path.exists():
        say("Bonjour.", path)
    y, sr = librosa.load(path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=80)
    log_mel = librosa.power_to_db(mel)
    print(log_mel.shape)  # (80, number of frames): 100 frames per second


if __name__ == "__main__":
    main()
