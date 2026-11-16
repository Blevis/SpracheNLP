"""Sentence and word tokenisation with NLTK (Punkt, German model) plus fixes for German.

NLTK's German Punkt model does not know a few abbreviations that are common in
Wikipedia ("z. B.", "u. a.", "Mio.") and Treebank tokenisation keeps number
ranges like "1914–1918" together. Both are fixed here so that every later step
(BIO labels, chunk grammars, CRF) sees one token per date, number and symbol.
"""
from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.tokenize import PunktTokenizer, word_tokenize

# lower-cased, without the final period, as Punkt stores them
EXTRA_ABBREVIATIONS = {
    "z", "b", "z.b", "u", "a", "u.a", "d", "h", "d.h", "bzw", "ca", "vgl", "evtl", "ggf", "usw", "sog",
    "mio", "mrd", "tsd", "nr", "st", "dr", "prof", "hl", "jh", "jhd", "v", "chr", "n", "s", "o", "geb",
    "gest", "km", "abk", "eigtl", "lat", "griech", "engl", "franz", "ital", "span", "inkl", "max", "min",
}

_RANGE = re.compile(r"^(\d{1,4}\.?)([–—-])(\d{1,4}\.?)$")     # 1914–1918, 9.–11.
_QUOTES = {"``": '"', "''": '"'}                               # undo Treebank's quote rewriting


@lru_cache(maxsize=1)
def _punkt() -> PunktTokenizer:
    try:
        nltk.data.find("tokenizers/punkt_tab/german")
    except LookupError:                       # first run on a new machine (needs internet once)
        nltk.download("punkt_tab", quiet=True)
    tok = PunktTokenizer("german")
    tok._params.abbrev_types.update(EXTRA_ABBREVIATIONS)
    return tok


def sentences(text: str) -> list[str]:
    return _punkt().tokenize(text)


def words(sentence: str) -> list[str]:
    out: list[str] = []
    for w in word_tokenize(sentence, language="german", preserve_line=True):
        w = _QUOTES.get(w, w)
        m = _RANGE.match(w)
        if m:
            out += [m.group(1), m.group(2), m.group(3)]
        else:
            out.append(w)
    return out


def tokenize(text: str) -> list[list[str]]:
    """Text -> list of sentences -> list of word tokens."""
    return [words(s) for s in sentences(text)]
