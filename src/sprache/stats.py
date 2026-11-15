"""Corpus statistics on spaCy Docs.

Definitions (used consistently in the notebook and the table row):
* token       - every spaCy token except whitespace (words, numbers, punctuation)
* word token  - a token with ``is_alpha`` (letters only), lower-cased
* unique words (types) - distinct word tokens
* stop-word share - word tokens in spaCy's German stop-word list / word tokens
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean

import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc


def parse(nlp: Language, texts: dict[str, str], batch_size: int = 32) -> dict[str, Doc]:
    names = list(texts)
    return dict(zip(names, nlp.pipe((texts[n] for n in names), batch_size=batch_size)))


@dataclass
class CorpusStats:
    texts: int
    tokens: int
    word_tokens: int
    unique_words: int
    unique_lemmas: int
    sentences: int
    stopword_share: float
    words_per_text: float
    tokens_per_sentence: float
    word_freq: Counter = field(repr=False)
    content_freq: Counter = field(repr=False)

    @property
    def ttr(self) -> float:
        return self.unique_words / self.word_tokens if self.word_tokens else 0.0

    def top(self, n: int = 10, content_only: bool = False) -> list[tuple[str, int]]:
        return (self.content_freq if content_only else self.word_freq).most_common(n)

    def overview(self) -> pd.DataFrame:
        rows = [
            ("Texts", self.texts),
            ("Tokens (all, incl. punctuation/numbers)", self.tokens),
            ("Word tokens (alphabetic)", self.word_tokens),
            ("Unique words (types, lower-cased)", self.unique_words),
            ("Unique lemmas", self.unique_lemmas),
            ("Type-token ratio", round(self.ttr, 3)),
            ("Sentences", self.sentences),
            ("Avg. word tokens per text", round(self.words_per_text, 1)),
            ("Avg. tokens per sentence", round(self.tokens_per_sentence, 1)),
            ("Stop-word share of word tokens", f"{self.stopword_share:.1%}"),
        ]
        return pd.DataFrame(rows, columns=["measure", "value"]).set_index("measure")


def compute(docs: dict[str, Doc], stop_words: set[str]) -> CorpusStats:
    words, lemmas, content = [], set(), Counter()
    n_tokens = n_sents = 0
    per_text = []
    for doc in docs.values():
        toks = [t for t in doc if not t.is_space]
        w = [t for t in toks if t.is_alpha]
        n_tokens += len(toks)
        n_sents += sum(1 for _ in doc.sents)
        per_text.append(len(w))
        words += [t.lower_ for t in w]
        lemmas |= {t.lemma_.lower() for t in w}
        content.update(t.lower_ for t in w if t.lower_ not in stop_words)
    freq = Counter(words)
    stop = sum(c for wd, c in freq.items() if wd in stop_words)
    return CorpusStats(
        texts=len(docs), tokens=n_tokens, word_tokens=len(words), unique_words=len(freq),
        unique_lemmas=len(lemmas), sentences=n_sents,
        stopword_share=stop / len(words) if words else 0.0,
        words_per_text=mean(per_text) if per_text else 0.0,
        tokens_per_sentence=n_tokens / n_sents if n_sents else 0.0,
        word_freq=freq, content_freq=content)


def top_table(st: CorpusStats, n: int = 10, stop_words: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for rank, ((w, c), (cw, cc)) in enumerate(zip(st.top(n), st.top(n, content_only=True)), 1):
        rows.append(dict(rank=rank, word=w, count=c, share=f"{c / st.word_tokens:.1%}",
                         stop_word=bool(stop_words and w in stop_words),
                         content_word=cw, content_count=cc))
    return pd.DataFrame(rows).set_index("rank")


def per_topic(docs: dict[str, Doc], topic_of: dict[str, str], stop_words: set[str]) -> pd.DataFrame:
    groups: dict[str, dict[str, Doc]] = {}
    for name, d in docs.items():
        groups.setdefault(topic_of.get(name, "?"), {})[name] = d
    rows = []
    for topic, g in sorted(groups.items()):
        s = compute(g, stop_words)
        rows.append(dict(topic=topic, texts=s.texts, word_tokens=s.word_tokens,
                         unique_words=s.unique_words, ttr=round(s.ttr, 3),
                         top_content_words=", ".join(w for w, _ in s.top(5, content_only=True))))
    return pd.DataFrame(rows).set_index("topic")


def compare(a: CorpusStats, b: CorpusStats, labels=("raw", "clean")) -> pd.DataFrame:
    df = pd.concat([a.overview()["value"], b.overview()["value"]], axis=1)
    df.columns = list(labels)
    return df
