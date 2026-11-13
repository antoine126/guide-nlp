"""Tested regular expressions shared by several chapters (chapter 5)."""

import re

DAY = r"(?:0?[1-9]|[12]\d|3[01])"
MONTH = r"(?:0?[1-9]|1[0-2])"
YEAR = r"(?:19|20)\d{2}"
SEP = r"[/.-]"

NUMERIC_DATE = re.compile(rf"\b(?P<d>{DAY}){SEP}(?P<m>{MONTH}){SEP}(?P<y>{YEAR})\b")
