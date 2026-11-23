"""Local web app: the search engine plus a live walk-through of the project.

    python -m sprache serve            -> http://127.0.0.1:8000

Standard library only (http.server), so there is nothing extra to install.
Views (static/index.html): Present (slides with live numbers), Search,
Pipeline (any text through cleaning, spaCy, index terms and NER), Evaluation,
Corpus (every text with its entities and source).

JSON API
    GET  /api/overview                 corpus, cleaning and NER numbers for the slides
    GET  /api/search?q=..&topic=..&system=..   results + how the query was read
    GET  /api/doc?file=text001.txt     one text with entity spans and source data
    GET  /api/eval                     ablation tables and per-query results
    POST /api/analyse  {"text": "...", "noisy": false}
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import Counter
from functools import cached_property
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import config, corpus

STATIC = Path(__file__).parent / "static"
MAX_INPUT = 5000                      # characters accepted by /api/analyse
TYPES = {"text/html": ".html", "text/css": ".css", "application/javascript": ".js", "image/svg+xml": ".svg"}

# Tutorial 2 results (entity-level F1 on the 8 hand-checked gold texts), as reported in
# notebooks/02_named_entities.ipynb section 4 and the README. Recomputing them takes minutes
# (`python -m sprache ner-eval`), so the slides quote them instead.
NER_F1 = [
    {"system": "spaCy de_core_news_md alone", "ENAMEX": 0.35, "NUMEX": None, "TIMEX": None, "all": 0.29},
    {"system": "Our pipeline (CRF + knowledge base + rules)", "ENAMEX": 0.80, "NUMEX": 0.95, "TIMEX": 1.00, "all": 0.85},
]


class App:
    """Everything the handlers need, loaded once (about 15 s: spaCy model + index)."""

    def __init__(self):
        from ..analysis import load_model
        from ..search.engine import SearchEngine
        t = time.time()
        self.nlp = load_model()
        self.engine = SearchEngine(self.nlp)
        self.load_seconds = round(time.time() - t, 1)
        self._lock = threading.Lock()
        self._systems = None
        self._eval = None

    # ------------------------------------------------------------------ lazy parts
    @cached_property
    def ner(self):
        from ..ner.pipeline import NERPipeline
        from ..ner.tagger import EntityTagger
        if not (config.NER_MODELS / "enamex.crf.tagger").exists():
            return None
        return NERPipeline(self.nlp, self.engine.kb, EntityTagger(self.engine.kb).load(),
                           normaliser=self.engine.normaliser)

    def systems(self):
        with self._lock:
            if self._systems is None:
                from ..search.evaluate import systems
                self._systems = systems(self.nlp, self.engine.docs, self.engine.kb, self.engine.normaliser)
                self._systems[-1] = (self._systems[-1][0], self.engine)
            return self._systems

    # ------------------------------------------------------------------ endpoints
    @cached_property
    def overview(self) -> dict:
        from ..clean import clean_text
        raw = corpus.load(config.CORPUS_RAW)
        sources = corpus.read_sources()
        words = [int(r["words"]) for r in sources]
        rules: Counter = Counter()
        changed = 0
        for t in raw.values():
            c: Counter = Counter()
            clean_text(t, c)
            rules += c
            changed += bool(c)
        docs = self.engine.docs.values()
        ents = [e for d in docs for e in d.entities if e["layer"] == "outer"]
        top = Counter((e["canonical"], e["type"]) for e in ents if e["type"] in config.ENAMEX and e["qid"])
        idx = self.engine.index
        return {
            "texts": len(sources), "words": sum(words), "min_words": min(words), "max_words": max(words),
            "median_words": statistics.median(words), "licence": config.LICENCE,
            "retrieved": max(r["retrieved"] for r in sources),
            "topics": Counter(r["topic"] for r in sources).most_common(),
            "cleaning": {"changed": changed, "rules": rules.most_common()},
            "model": config.SPACY_MODEL,
            "entities": {"mentions": len(ents), "linked": sum(1 for e in ents if e["qid"]),
                         "by_type": Counter(e["type"] for e in ents).most_common(),
                         "top": [{"name": n, "type": t, "count": c} for (n, t), c in top.most_common(12)]},
            "index": {"terms": sum(1 for t in idx.postings if not t.startswith("@")),
                      "entity_terms": sum(1 for t in idx.postings if t.startswith("@")),
                      "compounds_split": sum(1 for d in docs for t in d.toks
                                             if t.pos in ("NOUN", "ADJ") and self.engine.splitter.split(_fold(t.lemma)))},
            "ner_f1": NER_F1,
            "documents": [{"file": d.file, "title": d.title, "topic": d.topic} for d in docs],
            "load_seconds": self.load_seconds,
        }

    def search(self, q: str, topic: str | None, system: int | None, k: int = 10) -> dict:
        engine = self.engine
        name = "full system"
        if system is not None:
            name, engine = self.systems()[system]
        qa, results = engine.search(q, k, topic or None)
        return {"query": q, "system": name,
                "analysis": {"normalised": qa.normalised, "corrections": qa.corrections, "tokens": qa.tokens,
                             "entities": qa.entities,
                             "terms": sorted(({"term": t, "weight": w, "df": len(engine.index.postings.get(t, ())),
                                               "idf": round(engine.index.idf(t), 2)} for t, w in qa.terms.items()),
                                             key=lambda x: -x["idf"])},
                "results": results}

    def doc(self, file: str) -> dict:
        d = self.engine.docs.get(file)
        if d is None:
            raise KeyError(file)
        src = next(r for r in corpus.read_sources() if r["file"] == file)
        raw = (config.CORPUS_RAW / file).read_text(encoding="utf-8")
        return {"file": file, "title": d.title, "topic": d.topic, "url": d.url, "permalink": d.permalink,
                "revision_id": src["revision_id"], "retrieved": src["retrieved"], "licence": src["licence"],
                "words": src["words"], "cleaned": raw.strip() != d.text.strip(),
                "segments": entity_segments(d.text, d.entities)}

    def analyse(self, text: str, noisy: bool) -> dict:
        from ..clean import clean_text
        from ..ner.pipeline import mentions_table
        from ..search.text import tokens as search_tokens
        rules: Counter = Counter()
        clean = clean_text(text, rules)
        doc = self.nlp(clean)
        toks = search_tokens(doc)
        a = self.engine.analyzer
        terms: dict[int, list] = {}
        for i, term, w in a.terms(toks, self.engine.splitter, self.engine.names):
            terms.setdefault(i, []).append({"term": term, "weight": w})
        table = [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_, "morph": str(t.morph),
                  "dep": t.dep_, "head": t.head.text, "stop": t.is_stop, "terms": terms.get(k, [])}
                 for k, t in enumerate(x for x in doc if not x.is_space)]
        out = {"clean": clean, "rules": rules.most_common(), "tokens": table,
               "sentences": [s.text for s in doc.sents], "ner": None}
        if self.ner is not None:
            sents = self.ner(clean, normalise=noisy)
            norm = [self.engine.normaliser(s.words) for s in sents] if noisy else [None] * len(sents)
            out["ner"] = {
                "sentences": [{"words": s.words, "normalised": n, "outer": s.outer, "inner": s.inner}
                              for s, n in zip(sents, norm)],
                "mentions": mentions_table(sents).drop(columns=["file"]).fillna("").to_dict("records")}
        return out

    @cached_property
    def evaluation(self) -> dict:
        from ..search import evaluate as ev
        queries = ev.load_queries()
        systems = self.systems()
        overall, by_kind = ev.ablation(self.nlp, queries, systems)
        per_query = ev.per_query(systems[-1][1], queries)
        base = ev.per_query(systems[0][1], queries)
        per_query["baseline nDCG@10"] = base["nDCG@10"]
        return {"overall": overall.reset_index().to_dict("records"),
                "by_kind": by_kind.reset_index().to_dict("records"),
                "per_query": per_query.round(3).to_dict("records"),
                "queries": [{"id": q.id, "relevant": [{"file": f, "title": self.engine.docs[f].title, "grade": g}
                                                      for f, g in q.relevant.items()]} for q in queries],
                "systems": [n for n, _ in systems]}


def _fold(w: str) -> str:
    from ..ner.kb import fold
    return fold(w)


def entity_segments(text: str, entities: list[dict]) -> list[dict]:
    """Text split into [{text, type?, qid?, canonical?}] using the NLTK token offsets of the entity index."""
    from ..ner import text as ner_text
    offsets: list[list[tuple[int, int]]] = []
    pos = 0
    for words in ner_text.tokenize(text):
        row = []
        for w in words:
            j = text.find(w, pos)
            if j < 0 or j - pos > 20:
                row.append((-1, -1))
                continue
            row.append((j, j + len(w)))
            pos = j + len(w)
        offsets.append(row)
    spans = []
    for e in entities:
        if e["layer"] != "outer":
            continue
        s, a, b = int(e["sent"]), int(e["start"]), int(e["end"])
        if s >= len(offsets) or b > len(offsets[s]):
            continue
        start, end = offsets[s][a][0], offsets[s][b - 1][1]
        if start >= 0 and end > start:
            spans.append((start, end, e))
    spans.sort(key=lambda x: x[0])
    out, cur = [], 0
    for start, end, e in spans:
        if start < cur:
            continue
        if start > cur:
            out.append({"text": text[cur:start]})
        out.append({"text": text[start:end], "type": e["type"], "qid": e["qid"], "canonical": e["canonical"],
                    "reading": e["reading"]})
        cur = end
    if cur < len(text):
        out.append({"text": text[cur:]})
    return out


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SpracheNLP"

        def log_message(self, fmt, *args):          # quieter console: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):                           # health checks
            self.send_response(200 if urlparse(self.path).path in ("", "/", "/index.html") else 404)
            self.end_headers()

        def _json(self, data, status: int = 200):
            self._send(json.dumps(data, ensure_ascii=False, default=_jsonable).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/overview":
                    return self._json(app.overview)
                if u.path == "/api/search":
                    system = int(q["system"]) if q.get("system", "").isdigit() else None
                    return self._json(app.search(q.get("q", "")[:200], q.get("topic"), system))
                if u.path == "/api/doc":
                    return self._json(app.doc(q.get("file", "")))
                if u.path == "/api/eval":
                    return self._json(app.evaluation)
                return self._static(u.path)
            except KeyError:
                return self._json({"error": "not found"}, 404)
            except Exception as e:                          # show the error in the UI, keep serving
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            if urlparse(self.path).path != "/api/analyse":
                return self._json({"error": "not found"}, 404)
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 4 * MAX_INPUT)
                body = json.loads(self.rfile.read(n) or b"{}")
                text = str(body.get("text", ""))[:MAX_INPUT]
                return self._json(app.analyse(text, bool(body.get("noisy"))))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _static(self, path: str):
            name = "index.html" if path in ("", "/") else path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = next((t for t, ext in TYPES.items() if f.suffix == ext), "application/octet-stream")
            self._send(f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""))

    return Handler


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    print("Loading the spaCy model and building the index ...")
    app = App()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"Ready in {app.load_seconds} s. Open {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
