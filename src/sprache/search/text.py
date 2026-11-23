"""German text analysis for the search engine: tokens -> index terms.

The same ``Analyzer`` runs on the documents (at indexing time) and on the
query (at search time), so both sides end up in the same term space. Each
step can be switched off, which is how the evaluation measures what every
preprocessing step contributes:

    surface   lower-cased word forms, nothing else                    (baseline)
    stopwords drop spaCy's German stop words (der, und, ist, ...)
    lemmas    spaCy lemma instead of the word form (Flüsse -> Fluss)
    fold      case, umlaut and ß folding (Fußball = Fussball, Zürich = Zuerich)
    compounds split German compounds into known parts
              (Rechtschreibwörterbuch -> recht + schreiben + wörterbuch)

Entity terms (Wikidata ids from the NER index) are added by ``index.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..ner.kb import fold as _fold, strip_genitive

CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB", "ADV", "NUM", "X"}
SPLIT_POS = {"NOUN", "ADJ", "X"}                     # not names: Heidelberg is no Heidel-Berg
LINKING = ("", "s", "es", "n", "en", "e", "er")        # Fugenelemente: Bund-es-liga, Sonne-n-schein
HEAD_MIN = 5                                          # minimum length of a head found by back-off
MIN_PART = 4                                          # shorter parts give nonsense splits (Ber+lin)
COMPOUND_WEIGHT = 0.5                                 # a compound part counts half a full occurrence


@dataclass(frozen=True)
class Tok:
    """One spaCy token, reduced to what the search engine needs (picklable, model-free)."""
    text: str
    lemma: str
    pos: str
    stop: bool
    start: int          # character offsets in the document
    end: int
    sent: int           # sentence number


def tokens(doc) -> list[Tok]:
    sent_of = {}
    for k, s in enumerate(doc.sents):
        for t in s:
            sent_of[t.i] = k
    return [Tok(t.text, t.lemma_ or t.text, t.pos_, t.is_stop, t.idx, t.idx + len(t.text), sent_of.get(t.i, 0))
            for t in doc if not t.is_space]


def is_word(t: Tok) -> bool:
    return any(c.isalnum() for c in t.text)


class Decompounder:
    """Corpus-driven compound splitter.

    The vocabulary maps every word form of the corpus (folded, at least ``MIN_PART``
    letters) to its lemma, plus stems that occur as the first part of compounds:
    verb stems (``schreiben`` -> ``schreib``) and noun stems (``Weihnachten`` ->
    ``weihnacht``). A word is split into the fewest vocabulary parts, optionally
    joined by a linking element (Fugenelement); ties go to the split with the
    longest head, because the last part carries the meaning in German: a
    *Weihnachtsmarkt* is a *Markt*. Parts are returned as lemmas, so
    *Weihnachtsmarkt* contributes ``weihnachten`` and ``markt``.
    """

    def __init__(self, vocabulary: dict[str, str]):
        self.vocab = {w: c for w, c in vocabulary.items() if len(w) >= MIN_PART and w.isalpha()}
        self._split = lru_cache(maxsize=None)(self._split_uncached)

    @classmethod
    def from_tokens(cls, docs: list[list[Tok]], stop_words: set[str] = frozenset()) -> "Decompounder":
        vocab: dict[str, str] = {}
        stems: dict[str, str] = {}
        for toks in docs:
            for t in toks:
                if t.stop or not t.text.isalpha():
                    continue
                lemma = _fold(t.lemma)
                vocab.setdefault(lemma, lemma)
                vocab.setdefault(_fold(t.text), lemma)
                ends = ("en", "ern", "eln", "n") if t.pos == "VERB" else ("en", "n", "e") if t.pos == "NOUN" else ()
                for end in ends:
                    if lemma.endswith(end) and len(lemma) - len(end) >= MIN_PART:
                        stems.setdefault(lemma[: -len(end)], lemma)
                        break
        for k, v in stems.items():
            vocab.setdefault(k, v)
        for w in stop_words:
            vocab.pop(w, None)
        return cls(vocab)

    def _split_uncached(self, word: str) -> tuple[str, ...] | None:
        best = None
        for i in range(MIN_PART, len(word) - MIN_PART + 1):
            left = word[:i]
            if left not in self.vocab:
                continue
            for link in LINKING:
                rest = word[i:]
                if not rest.startswith(link):
                    continue
                right = rest[len(link):]
                if len(right) < MIN_PART:
                    continue
                tail = (right,) if right in self.vocab else self._split(right)
                if not tail:
                    continue
                cand = (left,) + tail
                # fewest parts, then longest head, then the shortest linking element
                # ('energie'+'wende' rather than 'energi'+'e'+'wende')
                key = (len(cand), -len(cand[-1]), len(word) - sum(map(len, cand)))
                if best is None or key < best[0]:
                    best = (key, cand)
        return best[1] if best else None

    def split(self, word: str) -> tuple[str, ...]:
        """Parts of a folded word as they appear in it, or () if it is not a compound of known words."""
        if len(word) < 2 * MIN_PART or not word.isalpha():
            return ()
        return self._split(word) or ()

    def parts(self, word: str) -> list[str]:
        """Index terms contributed by a compound: every part (as its lemma) and every
        longer head ('rechtschreibwoerterbuch' -> woerterbuch, schreibwoerterbuch, recht, schreiben)."""
        p = self.split(word)
        if not p:
            # back-off: a known head after an unknown modifier still counts
            # ('Rechtschreibwörterbuch' -> 'wörterbuch' even if 'rechtschreib' is unknown)
            for i in range(MIN_PART, len(word) - HEAD_MIN + 1):
                if word[i:] in self.vocab:
                    return [self.vocab[word[i:]]]
            return []
        heads = ["".join(p[i:]) for i in range(1, len(p) - 1)]
        return list(dict.fromkeys([self.vocab.get(x, x) for x in p] + heads))


@dataclass(frozen=True)
class Analyzer:
    """Token list -> weighted index terms. Each flag is one preprocessing step."""
    stopwords: bool = True
    lemmas: bool = True
    fold: bool = True
    compounds: bool = True
    name: str = "full"

    def norm(self, w: str) -> str:
        return _fold(w) if self.fold else w.lower()

    def term(self, t: Tok, names: set[str] = frozenset()) -> str:
        if not self.lemmas:
            return self.norm(t.text)
        # spaCy's lemmatizer damages names it takes for plural nouns (Beethoven -> Beethove);
        # a word the knowledge base knows as a name keeps its form, minus a genitive -s
        if t.text[:1].isupper() and names:
            for form in (strip_genitive(t.text), t.text):       # Deutschlands -> Deutschland
                if _fold(form) in names:
                    return self.norm(form)
        return self.norm(t.lemma)

    def compound_parts(self, t: Tok, splitter: Decompounder | None) -> list[str]:
        if not (self.compounds and splitter):
            return []
        out = []
        if "-" in t.text.strip("-"):                       # Berlin-Brandenburg, Fußball-Bund
            out += [self.norm(p) for p in t.text.split("-") if len(p) >= 2]
        if t.pos in SPLIT_POS:
            out += splitter.parts(_fold(t.lemma if self.lemmas else t.text))
        return [p for p in dict.fromkeys(out) if p]

    def terms(self, toks: list[Tok], splitter: Decompounder | None = None,
              names: set[str] = frozenset()) -> list[tuple[int, str, float]]:
        """(token position, term, weight) for every term a token contributes."""
        out = []
        for i, t in enumerate(toks):
            if not is_word(t) or (self.stopwords and (t.stop or t.pos not in CONTENT_POS)):
                continue
            term = self.term(t, names)
            if not term:
                continue
            out.append((i, term, 1.0))
            for part in self.compound_parts(t, splitter):
                if part != term:
                    out.append((i, part, COMPOUND_WEIGHT))
        return out


# The ablation steps of the evaluation, from the baseline to the full analyzer.
ANALYZERS = [
    Analyzer(stopwords=False, lemmas=False, fold=False, compounds=False, name="surface forms"),
    Analyzer(stopwords=True, lemmas=False, fold=False, compounds=False, name="+ stop words removed"),
    Analyzer(stopwords=True, lemmas=True, fold=False, compounds=False, name="+ lemmas"),
    Analyzer(stopwords=True, lemmas=True, fold=True, compounds=False, name="+ umlaut/case folding"),
    Analyzer(stopwords=True, lemmas=True, fold=True, compounds=True, name="+ compound splitting"),
]
