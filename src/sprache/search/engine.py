"""BM25 search over the German corpus, with an entity layer from Tutorial 2.

Indexing (once, at start-up)
    clean texts -> spaCy -> Tok lists -> Analyzer terms   (text.py)
    + title terms (weight TITLE_WEIGHT)
    + entity terms "@Q64" for every linked mention in data/ner/entities.csv
    -> inverted index: term -> {document: weighted term frequency}

Query processing (every search; each step is reported back to the UI)
    1. noise normalisation (Tutorial 2): truecasing, umlaut restoration, typo repair
       ('muenchen bir' -> 'München Bier')
    2. spaCy on the normalised query -> lemmas, POS, stop words
    3. the same Analyzer as the documents -> terms, incl. compound parts
    4. entity linking with the knowledge base: names -> Wikidata ids, so
       'Bundesrepublik', 'BRD' and 'Deutschland' all reach the item Q183
    5. BM25 ranking, optional topic filter, snippet with the matching words marked
"""
from __future__ import annotations

import csv
import hashlib
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .. import config, corpus
from ..ner.kb import KnowledgeBase, fold
from .text import ANALYZERS, Analyzer, Decompounder, Tok, is_word, tokens

TITLE_WEIGHT = 3.0          # a query word in the article title counts like three in the text
ENTITY_WEIGHT = 1.0         # weight of one linked mention of the query's entity
K1, B = 1.2, 0.75           # standard BM25 parameters
CACHE = config.NER_MODELS / "search_tokens.pkl"


@dataclass
class Document:
    file: str
    title: str
    topic: str
    url: str
    permalink: str
    text: str
    toks: list[Tok]
    title_toks: list[Tok]
    entities: list[dict] = field(default_factory=list)      # rows of entities.csv for this text


def parse_corpus(nlp, use_cache: bool = True) -> dict[str, Document]:
    """Clean corpus + metadata + entity index -> Documents (spaCy results cached on disk)."""
    texts = corpus.load(config.CORPUS_CLEAN)
    meta = {r["file"]: r for r in corpus.read_sources()}
    key = {f: hashlib.md5((t + meta[f]["title"]).encode("utf-8")).hexdigest() for f, t in texts.items()}
    cached = None
    if use_cache and CACHE.exists():
        try:
            with open(CACHE, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("key") != key:
                cached = None
        except Exception:
            cached = None
    if cached is None:
        files = sorted(texts)
        body = [tokens(d) for d in nlp.pipe(texts[f] for f in files)]
        head = [tokens(d) for d in nlp.pipe(meta[f]["title"] for f in files)]
        cached = {"key": key, "body": dict(zip(files, body)), "title": dict(zip(files, head))}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump(cached, fh)
    ents: dict[str, list[dict]] = defaultdict(list)
    if config.NER_OUTPUT.exists():
        with open(config.NER_OUTPUT, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ents[r["file"]].append(r)
    return {f: Document(f, meta[f]["title"], meta[f]["topic"], meta[f]["url"], meta[f]["permalink"], texts[f],
                        cached["body"][f], cached["title"][f], ents.get(f, []))
            for f in sorted(texts)}


class Index:
    """Inverted index + BM25 for one Analyzer configuration."""

    def __init__(self, docs: dict[str, Document], analyzer: Analyzer, splitter: Decompounder | None,
                 entities: bool = True, names: set[str] = frozenset()):
        self.docs, self.analyzer, self.splitter, self.use_entities = docs, analyzer, splitter, entities
        self.postings: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.positions: dict[str, dict[int, set[str]]] = {}       # file -> token index -> its terms
        self.length: dict[str, float] = {}
        for f, d in docs.items():
            tf: Counter = Counter()
            pos: dict[int, set[str]] = defaultdict(set)
            for i, term, w in analyzer.terms(d.toks, splitter, names):
                tf[term] += w
                pos[i].add(term)
            for _, term, w in analyzer.terms(d.title_toks, splitter, names):
                tf[term] += w * TITLE_WEIGHT
            if entities:
                for e in d.entities:
                    if e["qid"] and e["layer"] == "outer":
                        tf["@" + e["qid"]] += ENTITY_WEIGHT
            for term, n in tf.items():
                self.postings[term][f] = n
            self.positions[f] = pos
            self.length[f] = sum(tf.values())
        self.avg_len = sum(self.length.values()) / max(len(self.length), 1)
        self.n_docs = len(docs)

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) if n else 0.0

    def score(self, query_terms: dict[str, float], allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scores: Counter = Counter()
        for term, qw in query_terms.items():
            idf = self.idf(term)
            for f, tf in self.postings.get(term, {}).items():
                if allowed is not None and f not in allowed:
                    continue
                norm = K1 * (1 - B + B * self.length[f] / self.avg_len)
                scores[f] += qw * idf * tf * (K1 + 1) / (tf + norm)
        return scores.most_common()


@dataclass
class QueryAnalysis:
    original: str
    normalised: str
    corrections: list[tuple[str, str]]
    tokens: list[dict]                       # word, lemma, pos, stop, terms
    entities: list[dict]                     # text, qid, type, label
    terms: dict[str, float]                  # term -> query weight


class SearchEngine:
    def __init__(self, nlp, docs: dict[str, Document] | None = None, analyzer: Analyzer = ANALYZERS[-1],
                 entities: bool = True, normalise: bool = True, kb: KnowledgeBase | None = None,
                 normaliser=None):
        self.nlp = nlp
        self.docs = docs if docs is not None else parse_corpus(nlp)
        self.analyzer = analyzer
        self.splitter = Decompounder.from_tokens([d.toks for d in self.docs.values()],
                                                 {fold(w) for w in nlp.Defaults.stop_words})
        self.kb = kb if kb is not None else KnowledgeBase.load()
        self.use_entities, self.use_normalise = entities, normalise
        # names that keep their form instead of spaCy's lemma: one-word names and surnames
        self.names = {k for k in self.kb.index if " " not in k} | {
            fold(e.label.split()[-1]) for e in self.kb.entries.values() if e.type == "PER"}
        self.index = Index(self.docs, analyzer, self.splitter if analyzer.compounds else None,
                           entities, self.names)
        self._normaliser = normaliser

    # ------------------------------------------------------------------ query side
    @property
    def normaliser(self):
        if self._normaliser is None and self.use_normalise:
            from ..ner import noise, text as ner_text
            sents = [w for d in self.docs.values() for w in ner_text.tokenize(d.text)]
            sents += [d.title.split() for d in self.docs.values()]
            self._normaliser = noise.Normaliser(sents, self.kb)
        return self._normaliser

    def analyse_query(self, q: str) -> QueryAnalysis:
        words = q.split()
        corrections = []
        if self.use_normalise and words:
            fixed = self.normaliser(words)
            corrections = [(a, b) for a, b in zip(words, fixed) if a != b]
            words = fixed
        norm_q = " ".join(words)
        toks = tokens(self.nlp(norm_q))
        # The query is analysed like the documents, with a back-off chain for words the
        # index does not know:
        #   lemma -> word form: spaCy lemmatises a lone query word without context and
        #            sometimes wrongly ('Gebirge' -> 'Gebirg'); the form is then tried as is;
        #   -> compound parts: 'Habsburgerreich' is not in the corpus, so it is searched as
        #            'habsburger' + 'reich'. A known compound such as 'Hauptstadt' stays whole
        #            (splitting it would also match every 'Stadt').
        known = self.index.postings
        by_pos: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for i, term, w in self.analyzer.terms(toks, None, self.names):
            if term not in known and self.analyzer.lemmas:
                form = self.analyzer.norm(toks[i].text)
                if form in known:
                    term = form
            by_pos[i].append((term, w))
            if self.analyzer.compounds and term not in known:
                for part in self.analyzer.compound_parts(toks[i], self.splitter):
                    if part != term:
                        by_pos[i].append((part, 1.0))
        terms: dict[str, float] = defaultdict(float)
        for lst in by_pos.values():
            for term, w in lst:
                terms[term] = max(terms[term], w)
        entities = self._link(toks) if self.use_entities else []
        for e in entities:
            terms["@" + e["qid"]] = max(terms["@" + e["qid"]], 1.0)
        tok_info = [{"word": t.text, "lemma": t.lemma, "pos": t.pos, "stop": t.stop,
                     "terms": [f"{a}" + ("" if w == 1 else f" (x{w:g})") for a, w in by_pos.get(i, [])]}
                    for i, t in enumerate(toks)]
        return QueryAnalysis(q, norm_q, corrections, tok_info, entities, dict(terms))

    def _link(self, toks: list[Tok]) -> list[dict]:
        words = [t.text for t in toks]
        out, i = [], 0
        while i < len(words):
            n = self.kb.longest_match(words, i) if words[i][:1].isupper() else 0
            if n:
                name = " ".join(words[i:i + n])
                cands = self.kb.candidates(name)
                if cands:
                    e = cands[0]
                    out.append({"text": name, "qid": e.qid, "type": e.type, "label": e.label})
                i += n
            else:
                i += 1
        return out

    # ------------------------------------------------------------------ search
    def search(self, q: str, k: int = 10, topic: str | None = None) -> tuple[QueryAnalysis, list[dict]]:
        qa = self.analyse_query(q)
        allowed = {f for f, d in self.docs.items() if d.topic == topic} if topic else None
        ranked = self.index.score(qa.terms, allowed)[:k]
        return qa, [self._result(f, s, qa) for f, s in ranked]

    def ranking(self, q: str, k: int = 10) -> list[str]:
        return [f for f, _ in self.index.score(self.analyse_query(q).terms)[:k]]

    def _result(self, f: str, score: float, qa: QueryAnalysis) -> dict:
        d = self.docs[f]
        matched = sorted({t for t in qa.terms if f in self.index.postings.get(t, {})})
        qids = {t[1:] for t in qa.terms if t.startswith("@")}
        ents = Counter((e["canonical"], e["type"], e["qid"]) for e in d.entities if e["layer"] == "outer"
                       and e["type"] in config.ENAMEX)
        return {"file": f, "title": d.title, "topic": d.topic, "url": d.url, "permalink": d.permalink,
                "score": round(score, 2), "matched": matched,
                "snippet": self.snippet(d, qa.terms, qids),
                "entities": [{"name": n, "type": t, "qid": q, "count": c} for (n, t, q), c in ents.most_common(6)]}

    def hit_tokens(self, d: Document, terms: dict[str, float], qids: set[str]) -> set[int]:
        pos = self.index.positions[d.file]
        hits = {i for i, ts in pos.items() if ts & terms.keys()}
        for e in d.entities:
            if e["qid"] in qids:
                hits |= self._find_span(d.toks, e["text"])
        return hits

    @staticmethod
    def _find_span(toks: list[Tok], text: str) -> set[int]:
        """Token positions of an entity mention. The entity index uses NLTK tokens and
        NLTK sentence numbers, so the mention is found again by its words."""
        words = text.split()
        out: set[int] = set()
        for j in range(len(toks) - len(words) + 1):
            if [toks[i].text for i in range(j, j + len(words))] == words:
                out |= set(range(j, j + len(words)))
        return out

    def snippet(self, d: Document, terms: dict[str, float], qids: set[str], max_words: int = 45) -> list[dict]:
        """Best sentence (most query weight), as [{text, hit}] segments for safe rendering."""
        hits = self.hit_tokens(d, terms, qids)
        by_sent: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(d.toks):
            by_sent[t.sent].append(i)
        pos = self.index.positions[d.file]

        def weight(s: int) -> float:
            return sum(max((self.index.idf(x) for x in pos.get(i, ()) if x in terms), default=0) + (1 if i in hits else 0)
                       for i in by_sent[s])
        best = max(by_sent, key=lambda s: (weight(s), -s)) if hits else 0
        idx = by_sent[best]
        if len(idx) > max_words:                      # long sentence: window around the first hit
            first = next((k for k, i in enumerate(idx) if i in hits), 0)
            lo = max(0, first - max_words // 3)
            idx = idx[lo:lo + max_words]
        segs, start = [], d.toks[idx[0]].start
        lead = "… " if best > 0 or start > d.toks[by_sent[best][0]].start else ""
        if lead:
            segs.append({"text": lead, "hit": False})
        prev_end = start
        for i in idx:
            t = d.toks[i]
            if t.start > prev_end:
                segs.append({"text": d.text[prev_end:t.start], "hit": False})
            segs.append({"text": t.text, "hit": i in hits and is_word(t)})
            prev_end = t.end
        if idx[-1] != by_sent[best][-1] or best < max(by_sent):
            segs.append({"text": " …", "hit": False})
        return segs
