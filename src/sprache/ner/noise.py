"""Noisy text: simulate it, and normalise it back before tagging.

Social-media and OCR-like German breaks the two strongest NER cues at once:
capitalisation (every noun is capitalised, names included) and spelling
(``Muenchen``, ``Muenchen``, ``muenchen``, typos). ``corrupt`` produces such
text from clean sentences, token by token, so the gold BIO labels still align.

``Normaliser`` undoes it with resources we already have, no extra data:
* a case/spelling lexicon from the corpus plus the KB names: folded form ->
  most frequent real spelling (``muenchen`` -> ``München``, ``berlin`` -> ``Berlin``);
  this is truecasing and umlaut restoration in one lookup;
* for unknown words, the nearest lexicon entry within edit distance 1
  (``nltk.edit_distance``), restricted to entries with the same first letter
  and similar length, and only if that entry is unambiguous.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from nltk import edit_distance

from .kb import KnowledgeBase, fold

ASCII = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
KINDS = ("lower", "ascii", "typo", "social")


def _typo(w: str, rng: random.Random) -> str:
    if len(w) < 5 or not w.isalpha():
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "drop":
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def corrupt(words: list[str], kind: str, seed: int = 0, rate: float = 0.15) -> list[str]:
    """Token-wise corruption. kind: lower | ascii | typo | social (= lower + ascii + typos)."""
    rng = random.Random(seed)
    out = []
    for w in words:
        if kind in ("ascii", "social"):
            w = "".join(ASCII.get(c, c) for c in w)
        if kind in ("lower", "social"):
            w = w.lower()
        if kind in ("typo", "social") and rng.random() < rate:
            w = _typo(w, rng)
        out.append(w)
    return out


class Normaliser:
    def __init__(self, sentences: list[list[str]], kb: KnowledgeBase | None = None):
        forms: dict[str, Counter] = defaultdict(Counter)
        for words in sentences:
            for i, w in enumerate(words):
                if w.isalpha() or "-" in w:
                    # sentence-initial capitals say nothing about the word's real case
                    forms[fold(w)][w] += 0.2 if i == 0 and w[:1].isupper() else 1.0
        if kb:
            for e in kb.entries.values():
                for n in e.names:
                    for w in n.split():
                        forms[fold(w)][w] += 0.5
        self.lexicon = {k: c.most_common(1)[0][0] for k, c in forms.items()}
        self.by_prefix: dict[str, list[str]] = defaultdict(list)
        for k in self.lexicon:
            if len(k) >= 5:
                self.by_prefix[k[:1]].append(k)

    def word(self, w: str) -> str:
        if not (w.isalpha() or "-" in w):
            return w
        key = fold(w)
        if key in self.lexicon:
            best = self.lexicon[key]
            # keep a capital the writer chose (sentence start); restore one they dropped
            return best if not w[:1].isupper() or best[:1].isupper() else w
        if len(key) >= 5:
            near = [k for k in self.by_prefix.get(key[:1], ()) if abs(len(k) - len(key)) <= 1
                    and edit_distance(k, key, transpositions=True) == 1]
            if len(near) == 1:
                return self.lexicon[near[0]]
        return w

    def __call__(self, words: list[str]) -> list[str]:
        return [self.word(w) for w in words]
