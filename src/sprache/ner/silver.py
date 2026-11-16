"""Automatic ("silver") BIO annotation of the corpus by distant supervision.

ENAMEX labels come from the article's own links (``links.csv``): a linked name
gets the Wikidata type of its target. Wikipedia links a name only once, so the
label is *propagated* to every later mention of the same name in the text,
including its variants (label, Wikidata aliases, genitive form, and a
person's surname alone: "Johann Sebastian Bach ... Bach"). Names linked in
other articles are also matched, but only when they were really used as link
text somewhere (a Wikidata alias alone is not enough evidence).

NUMEX/TIMEX labels come from the chunk grammar in ``rules.py``.

Nested entities: inside every multi-token name, sub-spans that are themselves
KB names go to the inner layer ("Regierungsbezirk [Düsseldorf]").
"""
from __future__ import annotations

import csv
from collections import defaultdict

from .. import config, corpus
from . import bio, rules, text
from .context import is_nested_name, merge_rules
from .doc import Sentence, parse
from .kb import KnowledgeBase, fold, strip_genitive

MAX_NAME_TOKENS = 8


def doc_names(file: str, kb: KnowledgeBase) -> dict[str, str]:
    """Folded name -> type for every entity linked from this text, with variants."""
    names: dict[str, str] = {}
    with open(config.NER_LINKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["file"] != file or r["qid"] not in kb.entries:
                continue
            e = kb.entries[r["qid"]]
            variants = set(e.names)
            if r["anchor"][:1].isupper():
                variants.add(r["anchor"])
            if e.type == "PER":                       # surname alone: "Bach", "Goethe"
                variants |= {n.split()[-1] for n in list(variants) if len(n.split()) > 1 and len(n.split()[-1]) > 3}
            for v in variants:
                names.setdefault(fold(v), e.type)
    return names


def _match(words: list[str], i: int, local: dict[str, str], kb: KnowledgeBase) -> tuple[int, str]:
    """Longest name starting at words[i]: (length, type), (0, '') if none."""
    if not words[i][:1].isupper():
        return 0, ""
    for n in range(min(MAX_NAME_TOKENS, len(words) - i), 0, -1):
        if not words[i + n - 1][:1].isalnum():      # names do not end in punctuation
            continue
        name = " ".join(words[i:i + n])
        for key in (fold(name), fold(strip_genitive(name))):
            if key in local:
                return n, local[key]
        cands = kb.candidates(name)
        if cands and any(kb.anchors[(fold(name), c.qid)] or kb.anchors[(fold(strip_genitive(name)), c.qid)]
                         for c in cands):
            return n, cands[0].type
    return 0, ""


LOC_SUFFIXES = ("rhein", "see", "berg", "gebirge", "wald", "tal", "insel", "meer", "bach", "heide", "ebene",
                "europa", "asien", "afrika", "amerika", "alpen", "küste", "kanal", "platz", "straße", "kirche",
                "dom", "turm", "brücke", "schloss", "burg", "park", "garten", "wiese")


def place_type(name: str, kb: KnowledgeBase) -> str:
    """GPE or LOC for a place name spaCy found: the KB if it knows the name, else the
    shape of the word (rivers, lakes, mountains, buildings end in typical nouns)."""
    types = kb.types(name)
    if types & {"GPE", "LOC"}:
        return "GPE" if "GPE" in types else "LOC"
    return "LOC" if name.lower().rstrip("s").endswith(LOC_SUFFIXES) else "GPE"


def annotate_sentence(s: Sentence, local: dict[str, str], kb: KnowledgeBase) -> Sentence:
    """Silver labels for one sentence. If spaCy entities are present (``s.ents``),
    they fill the gaps that links leave (names nobody linked)."""
    w = s.words
    outer: list[bio.Span] = []
    i = 0
    while i < len(w):
        n, typ = _match(w, i, local, kb)
        if n:
            outer.append((i, i + n, typ))
            i += n
        else:
            i += 1
    if s.ents:
        taken = {k for a, b, _ in outer for k in range(a, b)}
        for a, b, lab in bio.bio_to_spans(s.ents):
            proper = not s.tags or "NE" in s.tags[a:b]        # spaCy also labels common nouns ("Stadt")
            if lab in ("PER", "ORG", "LOC") and proper and w[a][:1].isupper() and not taken & set(range(a, b)):
                outer.append((a, b, place_type(s.text(a, b), kb) if lab == "LOC" else lab))
    inner = []
    for s0, e0, t0 in outer:
        if e0 - s0 < 2 or t0 == "PER":        # person names do not contain other names
            continue
        for a in range(s0 + 1, e0):
            for b in range(e0, a, -1):
                if is_nested_name(w, s0, e0, a, b, kb):
                    n, typ = _match(w[a:b], 0, local, kb)
                    if n == b - a:
                        inner.append((a, b, typ))
                        break
    outer = merge_rules(outer, rules.spans(w))
    inner = [x for x in inner if any(o[0] <= x[0] and x[1] <= o[1] for o in outer)]
    o, inn = bio.layers(outer + inner)
    s.outer, s.inner = bio.spans_to_bio(len(w), o), bio.spans_to_bio(len(w), inn)
    return s


def annotate_corpus(kb: KnowledgeBase | None = None, files: list[str] | None = None, nlp=None) -> list[Sentence]:
    """Silver-annotate the clean corpus. With ``nlp`` the sentences are parsed first and
    spaCy's entities complement the link-based labels."""
    kb = kb or KnowledgeBase.load()
    texts = corpus.load(config.CORPUS_CLEAN)
    sents = [Sentence(f, words) for f in (files or sorted(texts)) for words in text.tokenize(texts[f])]
    if nlp is not None:
        parse(nlp, sents)
    local: dict[str, dict] = {}
    for s in sents:
        if s.file not in local:
            local[s.file] = doc_names(s.file, kb)
        annotate_sentence(s, local[s.file], kb)
    return sents


def by_file(sentences: list[Sentence]) -> dict[str, list[Sentence]]:
    d: dict[str, list[Sentence]] = defaultdict(list)
    for s in sentences:
        d[s.file].append(s)
    return d
