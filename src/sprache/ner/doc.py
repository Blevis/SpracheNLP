"""Sentence container, spaCy parsing of NLTK tokens, and CoNLL input/output.

CoNLL layout (one token per line, blank line between sentences, GermEval-style
two entity columns for nested entities):

    # file = text017.txt
    Johann      NE      B-PER   O
    Sebastian   NE      I-PER   O
    Bach        NE      I-PER   O
    ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spacy.tokens import Doc

from . import bio


@dataclass
class Sentence:
    file: str
    words: list[str]
    tags: list[str] = field(default_factory=list)      # STTS POS tags (spaCy)
    lemmas: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)      # TIGER dependency labels (spaCy)
    heads: list[int] = field(default_factory=list)     # index of the syntactic head
    ents: list[str] = field(default_factory=list)      # spaCy's own NER as BIO (PER/ORG/LOC/MISC), a feature
    outer: list[str] = field(default_factory=list)     # BIO, outer layer
    inner: list[str] = field(default_factory=list)     # BIO, nested layer

    def __post_init__(self):
        n = len(self.words)
        self.outer = self.outer or ["O"] * n
        self.inner = self.inner or ["O"] * n

    def spans(self, layer: str = "outer") -> list[bio.Span]:
        return bio.bio_to_spans(getattr(self, layer))

    def text(self, s: int, e: int) -> str:
        return " ".join(self.words[s:e])


def parse(nlp, sentences: list[Sentence], batch_size: int = 64) -> list[Sentence]:
    """Add spaCy POS, lemma and dependencies to sentences *without* re-tokenising.

    The NLTK tokens are handed to spaCy as a ready-made ``Doc``, so both tools
    refer to the same token indices. spaCy's own entity labels are kept in
    ``ents`` as one more *feature* for the CRF; they are never the output.
    """
    docs = (Doc(nlp.vocab, words=s.words) for s in sentences)
    for s, d in zip(sentences, nlp.pipe(docs, batch_size=batch_size)):
        s.tags = [t.tag_ for t in d]
        s.lemmas = [t.lemma_ for t in d]
        s.deps = [t.dep_ for t in d]
        s.heads = [t.head.i for t in d]
        s.ents = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ in ("B", "I") else "O" for t in d]
    return sentences


def write_conll(sentences: list[Sentence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines, last = [], None
    for s in sentences:
        if s.file != last:
            lines.append(f"# file = {s.file}")
            last = s.file
        tags = s.tags or ["_"] * len(s.words)
        lines += [f"{w}\t{t}\t{o}\t{i}" for w, t, o, i in zip(s.words, tags, s.outer, s.inner)]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_conll(path: Path) -> list[Sentence]:
    out: list[Sentence] = []
    file, rows = "", []

    def flush():
        if rows:
            w, t, o, i = map(list, zip(*rows))
            out.append(Sentence(file, w, tags=[] if t[0] == "_" else t, outer=o, inner=i))
            rows.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# file = "):
            flush()
            file = line[9:].strip()
        elif not line.strip():
            flush()
        else:
            rows.append(line.split("\t"))
    flush()
    return out
