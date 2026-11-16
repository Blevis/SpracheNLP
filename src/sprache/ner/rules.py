"""NUMEX and TIMEX with an NLTK cascaded chunk grammar (``nltk.RegexpParser``).

Numbers, dates and amounts have a closed, regular shape, so rules beat a
statistical model trained on little data. Each token first gets a *word
class* (YEAR, ORD, MONTH, CUR, UNIT, ...); the chunk grammar then combines
classes into entities:

    am  21.  März   1685  ->  (DATE 21./ORD März/MONTH 1685/YEAR)
    um  3,5  %             ->  (PERCENT 3,5/NUM %/PCT)
        12,4 Mrd.  Euro    ->  (MONEY 12,4/NUM Mrd./SCALE Euro/CUR)

Stage order resolves type ambiguity between classes: "2000 Euro" is MONEY
(not a year) because MONEY is chunked before DATE, and "1500 Einwohner" is
not a date because a four-digit number directly before a noun only counts as
a year when a temporal cue precedes it ("seit 1990 Mitglied").
"""
from __future__ import annotations

import re
from functools import lru_cache

from nltk import RegexpParser
from nltk.tree import Tree

from .bio import Span

MONTHS = {"januar", "jänner", "februar", "märz", "april", "mai", "juni", "juli", "august", "september",
          "oktober", "november", "dezember", "jan.", "feb.", "febr.", "mär.", "apr.", "jun.", "jul.",
          "aug.", "sep.", "sept.", "okt.", "nov.", "dez."}
WEEKDAYS = {"montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonnabend", "sonntag"}
CENTURY = {"jahrhundert", "jahrhunderts", "jahrhunderten", "jahrtausend", "jahrtausends", "jh.", "jhd."}
DECADE_NOUN = {"jahre", "jahren", "jahrzehnt", "jahrzehnts"}
MODIFIER = {"anfang", "mitte", "ende", "beginn", "frühjahr", "sommer", "herbst", "winter"}
SCALE = {"tausend", "tsd.", "million", "millionen", "mio.", "mio", "milliarde", "milliarden", "mrd.", "mrd",
         "billion", "billionen", "hundert"}
CURRENCY = {"euro", "€", "eur", "dollar", "us-dollar", "$", "usd", "mark", "d-mark", "reichsmark", "franken",
            "pfund", "£", "yen", "cent", "gulden", "schilling", "rubel", "yuan", "kronen", "taler"}
PERCENT = {"%", "prozent", "prozentpunkte", "‰", "promille", "volumenprozent", "vol.-%", "gewichtsprozent"}
TIME_ADJ = {"frühen", "späten", "frühe", "späte", "ausgehenden", "beginnenden", "mittleren", "ersten", "zweiten"}
COUNT_NOUNS = {"einwohner", "einwohnern", "menschen", "personen", "ew."}
UNITS = {"km", "km²", "km2", "quadratkilometer", "quadratkilometern", "m", "meter", "metern", "kilometer",
         "kilometern", "m²", "quadratmeter", "hektar", "ha", "kg", "kilogramm", "tonnen", "t", "l", "liter",
         "cm", "mm", "°c", "°", "grad", "km/h", "mw", "kw", "gw", "ps", "m³", "kubikmeter", "meilen",
         "seemeilen", "ns", "µm", "nm", "mb", "gb", "tb", "einwohner/km²", "ew./km²", "lichtjahre"}
NUMBER_WORDS = {"zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf",
                "zwanzig", "dreißig", "vierzig", "fünfzig", "hundert", "tausend", "einer", "eine", "einem"}
DATE_CUES = {"jahr", "jahre", "jahres", "seit", "bis", "von", "ab", "um", "im", "in", "anno", "vor", "nach",
             "zwischen", "(", "–", "-", "sommer", "herbst", "winter", "frühjahr", "ende", "anfang", "mitte"}

_YEAR = re.compile(r"^(1\d{3}|20\d{2})$")
_DECADE = re.compile(r"^(1\d|20)?\d0er(-Jahre[n]?)?$")
_ORD = re.compile(r"^\d{1,2}\.$")
_NUM = re.compile(r"^[+\-−]?\d+([.,]\d+)*$")
_CLOCK = re.compile(r"^([01]?\d|2[0-3])[:.][0-5]\d$")

GRAMMAR = r"""
TIME:     {<CLOCK><UHR>?}
          {<NUM|ORD><UHR>}
MONEY:    {<NUM|YEAR>(<DASH><NUM|YEAR>)?<SCALE>?<CUR>}
          {<CUR><NUM|YEAR><SCALE>?}
PERCENT:  {<NUM|YEAR>(<DASH><NUM|YEAR>)?<PCT>}
QUANTITY: {<NUM|YEAR>(<DASH><NUM|YEAR>)?<SCALE>?<UNIT>}
          {<NUM|YEAR><CNT><PRO><UNIT>}
          {<NUM|YEAR|NUMW>(<DASH><NUM|YEAR>)?<SCALE>}
DATE:     {(<MOD><DET>?)?<TADJ>?<ORD>(<DASH|CONJ><TADJ>?<ORD>)?<CENT>}
          {<WEEKDAY>?<MOD>?<ORD>?<MONTH><YEAR>?}
          {<DECADE><DECN>?}
          {<YEAR><DASH><YEAR>}
          {<YEAR>}
"""


@lru_cache(maxsize=1)
def parser() -> RegexpParser:
    return RegexpParser(GRAMMAR)


def word_class(words: list[str], i: int) -> str:
    """The word class of words[i] that the chunk grammar works on."""
    w = words[i]
    lw = w.lower()
    nxt = words[i + 1] if i + 1 < len(words) else ""
    prev = words[i - 1].lower() if i else ""
    if _CLOCK.match(w) and nxt.lower() == "uhr":
        return "CLOCK"
    if lw == "uhr":
        return "UHR"
    if prev in MONTHS and re.fullmatch(r"\d{1,4}", w):              # "2. Februar 962"
        return "YEAR"
    if _YEAR.match(w):
        # "1500 Einwohner": a number before a noun is a count, unless a temporal cue precedes it
        if nxt[:1].isupper() and nxt.lower() not in MONTHS and prev not in DATE_CUES:
            return "NUM"
        return "YEAR"
    if _DECADE.match(w):
        return "DECADE"
    if _ORD.match(w):
        return "ORD"
    if _NUM.match(w):
        return "NUM"
    if lw in MONTHS:
        # a bare month is a date only after a temporal cue or with a day/year:
        # "im August", "Mitte September", "3. Oktober" - but not "Herzog Carl August"
        if prev in DATE_CUES or prev in MODIFIER or _ORD.match(words[i - 1] if i else "") or _YEAR.match(nxt):
            return "MONTH"
        return "O"
    for cls, vocab in (("MONTH", MONTHS), ("WEEKDAY", WEEKDAYS), ("CENT", CENTURY), ("DECN", DECADE_NOUN),
                       ("SCALE", SCALE), ("CUR", CURRENCY), ("PCT", PERCENT), ("UNIT", UNITS)):
        if lw in vocab:
            return cls
    if lw in MODIFIER:
        return "MOD"
    if lw == "des":
        return "DET"
    if lw in TIME_ADJ:
        return "TADJ"
    if lw in ("und", "oder"):
        return "CONJ"
    if lw in COUNT_NOUNS:
        return "CNT"
    if lw == "pro":
        return "PRO"
    if w in ("–", "-", "—", "bis"):
        return "DASH"
    if lw in NUMBER_WORDS:
        return "NUMW"
    return "O"


def chunk(words: list[str]) -> Tree:
    """NLTK chunk tree of NUMEX/TIMEX for one tokenised sentence."""
    return parser().parse([(w, word_class(words, i)) for i, w in enumerate(words)])


def spans(words: list[str]) -> list[Span]:
    """NUMEX/TIMEX spans of one sentence (token offsets, end exclusive)."""
    out, i = [], 0
    for node in chunk(words):
        if isinstance(node, Tree):
            n = len(node.leaves())
            out.append((i, i + n, node.label()))
            i += n
        else:
            i += 1
    return out
