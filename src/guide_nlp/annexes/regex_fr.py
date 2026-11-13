"""Regular expressions for French text, ready to use (appendix A)."""

WORD = r"[^\W\d_]+(?:['’-][^\W\d_]+)*"  # L'été, Saint-Étienne, aujourd'hui
ELISION = r"(?i)\b(?:[cdjlmnst]|qu|jusqu|lorsqu|puisqu)['’]"  # L', qu', jusqu’
DATE = r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(\d{4})\b"  # 03/11/2026
MONTHS = "janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
LONG_DATE = rf"\b\d{{1,2}}(?:er)? (?:{MONTHS}) \d{{4}}\b"  # 1er mai 2024
PHONE = r"(?:\+33[ .-]?|\b0)[1-9](?:[ .-]?\d{2}){4}\b"  # 01 23 45 67 89, +33 6 12 34 56 78
POSTCODE = r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b"  # 75011, 97400, pas 00123
AMOUNT = r"\b\d{1,3}(?:[   .]\d{3})*(?:,\d{1,2})?\s?(?:€|EUR\b|euros?\b)"  # 1 234,56 €
EMAIL = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # marie.durand+test@exemple.fr
URL = r"https?://[^\s<>\"')\]]+"  # s'arrête avant ) ou ]
SIREN = r"\b\d{3} ?\d{3} ?\d{3}\b"  # 552 100 554
IBAN = r"\bFR\d{2}(?: ?[A-Z0-9]{4}){5} ?[A-Z0-9]{3}\b"  # FR76 3000 6000 0112 3456 7890 189
SENTENCE_END = r"(?<=[.!?…])\s+(?=[A-ZÉÈÀÂÊÎ«])"  # à utiliser avec re.split
