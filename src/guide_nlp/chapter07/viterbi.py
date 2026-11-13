"""The Viterbi algorithm on a toy hidden Markov model (chapter 7).

Usage: uv run python -m guide_nlp.chapter07.viterbi
"""

import math
from collections.abc import Sequence

TAGS = ["DET", "PRON", "ADJ", "NOUN", "VERB"]
START = {"DET": 0.45, "PRON": 0.25, "ADJ": 0.05, "NOUN": 0.15, "VERB": 0.10}
TRANS = {  # P(next tag | previous tag)
    "DET":  {"DET": .01, "PRON": .01, "ADJ": .30, "NOUN": .65, "VERB": .03},
    "PRON": {"DET": .05, "PRON": .10, "ADJ": .05, "NOUN": .10, "VERB": .70},
    "ADJ":  {"DET": .05, "PRON": .05, "ADJ": .10, "NOUN": .75, "VERB": .05},
    "NOUN": {"DET": .15, "PRON": .10, "ADJ": .25, "NOUN": .10, "VERB": .40},
    "VERB": {"DET": .50, "PRON": .15, "ADJ": .10, "NOUN": .15, "VERB": .10},
}  # fmt: skip
EMIT = {  # P(word | tag), restricted to the words of the example
    "DET":  {"la": .20, "le": .20},
    "PRON": {"la": .05, "le": .05},
    "ADJ":  {"belle": .010, "ferme": .005},
    "NOUN": {"belle": .002, "ferme": .003, "voile": .002},
    "VERB": {"ferme": .004, "voile": .001},
}  # fmt: skip
UNSEEN = 1e-12  # emission probability of a word never seen with a tag


def viterbi(words: Sequence[str]) -> list[str]:
    """Most probable tag sequence under a first-order HMM (log space)."""

    def emit(tag: str, word: str) -> float:
        return math.log(EMIT[tag].get(word, UNSEEN))

    columns: list[dict[str, tuple[float, str | None]]] = [
        {t: (math.log(START[t]) + emit(t, words[0]), None) for t in TAGS}
    ]
    for word in words[1:]:
        previous, column = columns[-1], {}
        for tag in TAGS:
            back, score = max(
                ((p, previous[p][0] + math.log(TRANS[p][tag])) for p in TAGS),
                key=lambda pair: pair[1],
            )
            column[tag] = (score + emit(tag, word), back)
        columns.append(column)

    tag = max(TAGS, key=lambda t: columns[-1][t][0])
    path = [tag]
    for column in reversed(columns[1:]):
        back = column[tag][1]
        assert back is not None
        tag = back
        path.append(tag)
    return path[::-1]


if __name__ == "__main__":
    print(viterbi("la belle ferme le voile".split()))
    # ['DET', 'NOUN', 'VERB', 'DET', 'NOUN']
    print(viterbi("la belle ferme".split()))
    # ['DET', 'ADJ', 'NOUN']
