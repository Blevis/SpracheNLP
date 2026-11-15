"""Rule-based cleaning of Wikipedia plaintext artefacts.

Each rule is a (name, pattern, replacement) triple so that the notebook can
report how often every rule fired. Rules are conservative on purpose: they
only remove material that is clearly *not* running German text (IPA
transcriptions, labels whose content was stripped by the API, empty brackets,
broken whitespace). Greek/Hebrew etymologies and editorial brackets such as
"[seiner]" are kept.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from . import config
from .corpus import load

# Characters that only occur in IPA transcriptions, never in German words:
# IPA Extensions (U+0250-02AF), spacing modifiers incl. ˈ ˌ ː (U+02B0-02FF),
# the combining marks used for syllabic/non-syllabic/tie (̩ ̯ ͡) and ŋ.
# Deliberately NOT ç or χ: they also occur in French and Greek words.
IPA_CHARS = "\u0250-\u02af\u02b0-\u02ff\u0329\u032f\u0361\u014b"
IPA = re.compile(rf"[{IPA_CHARS}]")

# Phase 1: remove artefacts. Run to a fixed point *before* tidying, because
# the dangling-label rules rely on the whitespace gaps that tidying removes.
REMOVE_RULES: list[tuple[str, re.Pattern, str]] = [
    # [ˈdʏsl̩ˌdɔʁf]  -> ''   (only bracket groups that contain IPA characters)
    # Replaced by a space, not '', so the gap it leaves ("label ;") stays
    # visible to the dangling-label rules below.
    ("ipa_transcription", re.compile(rf"\s*\[[^\[\]]*[{IPA_CHARS}][^\[\]]*\]"), " "),
    # "(Aussprache )" / "deutsche Aussprache ;" left behind after the IPA is gone
    ("dangling_pronunciation_label",
     re.compile(r"\b(?:deutsche |standarddeutsche )?Aussprache\s*(?=[,;)])"), ""),
    # "französisch , italienisch ..." -> the foreign word was a stripped template.
    # Whitespace before the separator is the signature: plain Wikipedia text
    # never has a space before "," or ";".
    ("dangling_language_label",
     re.compile(r"\b[a-zäöü]+(?:sch|deutsch|lektal)\s+(?=[,;)])"), ""),
    # "standarddeutsch  oder ;" -> the conjunction joined two stripped items
    ("dangling_conjunction", re.compile(r"\b(?:oder|und|auch)\s+(?=[,;)])"), ""),
]

# Phase 2: tidy punctuation and whitespace around what was removed.
TIDY_RULES: list[tuple[str, re.Pattern, str]] = [
    # "( ; obersorbisch" -> "(obersorbisch",  "Hamborg ; )" -> "Hamborg)"
    ("bracket_separator", re.compile(r"\(\s*[,;]\s*"), "("),
    ("bracket_separator", re.compile(r"\s*[,;]\s*\)"), ")"),
    ("double_separator", re.compile(r"([,;])\s*[,;]"), r"\1"),
    ("empty_brackets", re.compile(r"\s*\(\s*\)"), ""),
    ("space_before_punct", re.compile(r"\s+(?=[,.;:)\]])"), ""),
    ("space_after_bracket", re.compile(r"(?<=\()\s+"), ""),
    ("multiple_spaces", re.compile(r"[ \t]{2,}"), " "),
]


def _apply(text: str, rules, stats: Counter | None) -> str:
    for _ in range(10):                     # rules can enable each other
        before = text
        for name, rx, repl in rules:
            text, n = rx.subn(repl, text)
            if stats is not None and n:
                stats[name] += n
        if text == before:
            break
    return text


def clean_text(text: str, stats: Counter | None = None) -> str:
    """Apply removal rules, then tidy rules; optionally count rule hits."""
    text = _apply(text, REMOVE_RULES, stats)
    return _apply(text, TIDY_RULES, stats).strip()


RULES = REMOVE_RULES + TIDY_RULES           # for documentation / reporting


def clean_corpus(src: Path = config.CORPUS_RAW, dst: Path = config.CORPUS_CLEAN) -> dict[str, Counter]:
    """Write cleaned copies of every text; return {file: Counter(rule hits)}."""
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("text*.txt"):
        old.unlink()
    report = {}
    for name, text in load(src).items():
        hits: Counter = Counter()
        (dst / name).write_text(clean_text(text, hits), encoding="utf-8")
        report[name] = hits
    return report
