"""Static export of the web app for GitHub Pages (or any static host).

    python -m sprache export-site            -> site/

GitHub Pages cannot run Python, so everything the server computes ahead of a
query is computed here once and written as JSON:

    data/search.json    inverted index (BM25 postings, lengths), the analyzer's resources
                        (word form -> lemma/POS table, stop words, names, compound
                        vocabulary), the normaliser lexicon and the knowledge-base index
    data/docs.json      every text: tokens, index terms per token, entity spans, metadata
    data/overview.json  numbers for the slides
    data/eval.json      the evaluation tables (computed in Python)
    data/pipeline.json  the Pipeline view for its example texts (spaCy and the CRF cannot
                        run in the browser, so free text needs `python -m sprache serve`)

``static-engine.js`` repeats the query side in JavaScript: normalisation, lemma
lookup, the same analyzer and back-off chain, entity linking, BM25 and snippets.
The one approximation: a query word's lemma and part of speech come from a table
of every word form in the corpus instead of from spaCy, so unseen word forms are
used as they are. ``check_parity`` compares both implementations on the test queries.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .. import config
from .server import STATIC, App, entity_segments

SITE_SRC = Path(__file__).parent / "site"
DEFAULT_OUT = config.ROOT / "site"

# example texts of the Pipeline view (must match PIPE_EXAMPLES in app.js)
PIPE_EXAMPLES = [
    "Düsseldorf ([ˈdʏsl̩ˌdɔʁf] , Düsseldorfer Platt Düsseldörp) ist die Landeshauptstadt Nordrhein-Westfalens und der Behördensitz des Regierungsbezirks Düsseldorf.",
    "Die Donaudampfschifffahrtsgesellschaft und der Weihnachtsmarkt sind typisch deutsche Komposita.",
    "Angela Merkel war von 2005 bis 2021 Bundeskanzlerin. Berlin lehnte den Vorschlag der Europäischen Union am 3. Oktober ab.",
    "gestern war ich in muenchen auf dem oktoberfest und habe bier fuer 15 euro getrunken",
]


def _dump(path: Path, data) -> int:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=_plain)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _plain(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def search_data(app: App) -> dict:
    eng = app.engine
    idx = eng.index
    files = list(eng.docs)
    pos_of = {f: i for i, f in enumerate(files)}
    postings = {t: {pos_of[f]: round(tf, 4) for f, tf in p.items()} for t, p in idx.postings.items()}

    # word form -> most frequent (lemma, POS) in the corpus, for query words
    forms: dict[str, Counter] = defaultdict(Counter)
    for d in eng.docs.values():
        for t in d.toks + d.title_toks:
            forms[t.text][(t.lemma, t.pos)] += 1
    lex = {w: list(c.most_common(1)[0][0]) for w, c in forms.items()}

    kb = eng.kb
    anchors: dict[str, dict[str, int]] = defaultdict(dict)
    for (key, qid), n in kb.anchors.items():
        anchors[key][qid] = n
    norm = eng.normaliser
    return {
        "files": files, "N": idx.n_docs, "avg_len": idx.avg_len, "length": [idx.length[f] for f in files],
        "postings": postings,
        "params": {"k1": 1.2, "b": 0.75, "title_weight": 3.0},
        "stop": sorted(eng.nlp.Defaults.stop_words),
        "lex": lex,
        "names": sorted(eng.names),
        "vocab": eng.splitter.vocab,
        "lexicon": norm.lexicon,
        "kb": {"index": {k: sorted(v) for k, v in kb.index.items()},
               "stem": {k: sorted(v) for k, v in kb.stem_index.items()},
               "anchors": anchors,
               "entries": {q: [e.type, e.label] for q, e in kb.entries.items()},
               "max_len": kb.max_len},
    }


def docs_data(app: App) -> list[dict]:
    eng = app.engine
    out = []
    for f, d in eng.docs.items():
        meta = app.doc(f)
        pos = eng.index.positions[f]
        ents = Counter((e["canonical"], e["type"], e["qid"]) for e in d.entities
                       if e["layer"] == "outer" and e["type"] in config.ENAMEX)
        ent_texts: dict[str, list[str]] = defaultdict(list)
        for e in d.entities:
            if e["qid"] and e["text"] not in ent_texts[e["qid"]]:
                ent_texts[e["qid"]].append(e["text"])
        out.append({
            **{k: meta[k] for k in ("file", "title", "topic", "url", "permalink", "revision_id", "retrieved",
                                    "licence", "words", "cleaned")},
            "segments": meta["segments"],
            "text": d.text,
            "toks": [[t.start, t.end, t.sent, int(any(c.isalnum() for c in t.text))] for t in d.toks],
            "title_words": [t.text for t in d.title_toks if any(c.isalnum() for c in t.text)],
            "terms": {i: sorted(ts) for i, ts in pos.items()},
            "entities": [{"name": n, "type": t, "qid": q, "count": c} for (n, t, q), c in ents.most_common(6)],
            "ent_texts": ent_texts,
        })
    return out


def pipeline_data(app: App) -> list[dict]:
    return [{"text": t, "noisy": noisy, "result": app.analyse(t, noisy)}
            for t in PIPE_EXAMPLES for noisy in (False, True)]


def export(out: Path = DEFAULT_OUT, app: App | None = None) -> dict[str, int]:
    """Write the static site to ``out`` (replacing it); return {file: bytes}."""
    app = app or App()
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    sizes = {}
    for name in ("app.js", "style.css"):
        shutil.copy2(STATIC / name, out / name)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="app.js"></script>',
                        '<script src="static-engine.js"></script>\n<script src="app.js"></script>')
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(SITE_SRC / "static-engine.js", out / "static-engine.js")
    (out / ".nojekyll").write_text("", encoding="utf-8")         # serve files as they are, no Jekyll
    sizes["data/search.json"] = _dump(out / "data" / "search.json", search_data(app))
    sizes["data/docs.json"] = _dump(out / "data" / "docs.json", docs_data(app))
    sizes["data/overview.json"] = _dump(out / "data" / "overview.json", app.overview)
    sizes["data/eval.json"] = _dump(out / "data" / "eval.json", app.evaluation)
    sizes["data/pipeline.json"] = _dump(out / "data" / "pipeline.json", pipeline_data(app))
    return sizes


def check_parity(out: Path = DEFAULT_OUT, app: App | None = None, k: int = 10) -> list[dict]:
    """Run the test queries through the JavaScript engine (Node.js) and the Python engine;
    return one row per query with both top-k rankings."""
    import subprocess
    from ..search.evaluate import load_queries
    app = app or App()
    queries = [q.text for q in load_queries()] + ["Flüsse in Deutschland", "muenchen bier", "Goethes Werke",
                                                  "Habsburgerreich", "Autohersteller", "Bundesrepublik"]
    script = SITE_SRC / "parity.js"
    res = subprocess.run(["node", str(script), str(out), json.dumps(queries, ensure_ascii=False), str(k)],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    js = json.loads(res.stdout)
    rows = []
    for q in queries:
        py = app.engine.ranking(q, k)
        rows.append({"query": q, "python": py, "js": js[q], "same_top1": py[:1] == js[q][:1],
                     "overlap": len(set(py) & set(js[q])) / max(len(py), 1)})
    return rows
