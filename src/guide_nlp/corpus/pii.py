"""Personal data detection and masking with Presidio (chapter 10).

Usage: uv run python -m guide_nlp.corpus.pii
"""

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

ENTITIES = ["PERSON", "FR_PHONE", "EMAIL", "LOCATION"]


def build_analyzer() -> AnalyzerEngine:
    """A French analyzer: spaCy for names and places, regexes for the rest."""
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "fr", "model_name": "fr_core_news_sm"}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["fr"])
    for name, regex, score in [
        ("FR_PHONE", r"(?:\+33\s?|\b0)[1-9](?:[\s.-]?\d{2}){4}\b", 0.7),
        ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 0.9),
    ]:
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity=name,
                supported_language="fr",
                patterns=[Pattern(name.lower(), regex, score)],
            )
        )
    return analyzer


def anonymize(text: str, analyzer: AnalyzerEngine) -> str:
    """Replace every detected entity by its type, e.g. ``<PERSON>``."""
    results = analyzer.analyze(text=text, language="fr", entities=ENTITIES)
    # presidio_analyzer and presidio_anonymizer each declare their own (identical)
    # RecognizerResult class: the objects are compatible, the annotations are not.
    anonymizer = AnonymizerEngine()
    return anonymizer.anonymize(text=text, analyzer_results=results).text  # ty: ignore[invalid-argument-type]


def main() -> None:
    text = (
        "Contactez Marie Dubois au 06 12 34 56 78 ou à marie.dubois@exemple.fr, "
        "12 rue Victor Hugo à Lyon."
    )
    print(anonymize(text, build_analyzer()))
    # Book: <PERSON> au <FR_PHONE> ou à <EMAIL>, 12 rue <PERSON> à <LOCATION>.


if __name__ == "__main__":
    main()
