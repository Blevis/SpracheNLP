"""Automatic quality checks for a corpus ({filename: text}).

All checks are plain functions of the text so they run without spaCy and can
be applied to the raw and the cleaned corpus alike for a before/after table.
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import pandas as pd

from .clean import IPA_CHARS
from .corpus import split_sentences

MOJIBAKE = re.compile(r"Ã[\x80-\xbf]|â€|�|Â[\x80-\xbf ]")

PATTERN_CHECKS: dict[str, re.Pattern] = {
    "mojibake (broken UTF-8)": MOJIBAKE,
    "HTML tags": re.compile(r"<[^>]+>"),
    "HTML entities": re.compile(r"&(?:[a-z]+|#\d+);"),
    "wiki markup (==, [[, {{)": re.compile(r"==|\[\[|\]\]|\{\{|\}\}"),
    "IPA transcription": re.compile(rf"\[[^\[\]]*[{IPA_CHARS}][^\[\]]*\]"),
    "empty brackets": re.compile(r"\(\s*[;,]?\s*\)"),
    "dangling label / space before , ;": re.compile(r"\S\s+[,;]"),
    "multiple spaces": re.compile(r" {2,}"),
    "non-Latin script": re.compile(
        r"[\u0370-\u03ff\u1f00-\u1fff\u0400-\u04ff\u0590-\u06ff\u0900-\u0dff\u3040-\u30ff\u4e00-\u9fff]+"),
    "boilerplate words": re.compile(
        r"\b(?:Cookie|Impressum|Navigation|Weiterleitung|Siehe auch|Einzelnachweise)\b"),  # case-sensitive: "(siehe auch ...)" in running text is fine
}

_EN_FUNCTION_WORDS = {"the", "and", "of", "is", "was", "with", "for", "from", "by", "at", "which"}


def _example(text: str, m: re.Match, width: int = 25) -> str:
    return "…" + text[max(0, m.start() - width): m.end() + width].replace("\n", " ") + "…"


def pattern_report(texts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for check, rx in PATTERN_CHECKS.items():
        files, hits, example = [], 0, ""
        for name, text in texts.items():
            ms = list(rx.finditer(text))
            if ms:
                files.append(name)
                hits += len(ms)
                example = example or f"{name}: {_example(text, ms[0])}"
        rows.append(dict(check=check, files_affected=len(files), occurrences=hits,
                         example=example, files=", ".join(files[:8]) + (" …" if len(files) > 8 else "")))
    return pd.DataFrame(rows)


def repeated_sentences(texts: dict[str, str], min_words: int = 5) -> list[tuple[str, int]]:
    c = Counter(s.strip() for t in texts.values() for s in split_sentences(t)
                if len(s.split()) >= min_words)
    return [(s, n) for s, n in c.items() if n > 1]


def english_sentences(texts: dict[str, str], threshold: int = 3) -> list[tuple[str, str]]:
    """Very rough: a sentence with >= threshold distinct English function words."""
    out = []
    for name, t in texts.items():
        for s in split_sentences(t):
            if len(_EN_FUNCTION_WORDS & set(re.findall(r"[a-z]+", s.lower()))) >= threshold:
                out.append((name, s))
    return out


def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def near_duplicates(texts: dict[str, str], threshold: float = 0.3) -> list[tuple[str, str, float]]:
    """Pairs of texts whose 5-word-shingle Jaccard similarity >= threshold."""
    sh = {n: _shingles(t) for n, t in texts.items()}
    out = []
    for a, b in combinations(sh, 2):
        if sh[a] and sh[b]:
            j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
            if j >= threshold:
                out.append((a, b, round(j, 3)))
    return sorted(out, key=lambda x: -x[2])


def max_similarity(texts: dict[str, str]) -> tuple[str, str, float]:
    """Most similar pair (for reporting even when nothing crosses the threshold)."""
    pairs = near_duplicates(texts, threshold=0.0)
    return pairs[0] if pairs else ("", "", 0.0)


def summary(texts: dict[str, str]) -> str:
    """One-line description of the problems found, for the comparison table."""
    rep = pattern_report(texts)
    parts = [f"{r.check}: {r.files_affected} file{'s' if r.files_affected != 1 else ''}"
             for r in rep.itertuples() if r.files_affected]
    return "; ".join(parts) or "none detected"
