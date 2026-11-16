"""A small entity knowledge base built from the corpus itself (distant supervision).

Nobody hand-labelled German NER training data for this project, so the labels
come from the encyclopedia: every Wikipedia intro links its important names to
their own articles, and Wikidata says what each article is about
(human -> PER, city -> GPE, company -> ORG, river -> LOC, ...).

``build()`` (network, about 1 minute, run once and committed like the corpus)
1. reads the wikitext of the *exact revision* of each intro we stored
   (``sources.csv``) and collects every article link: anchor text -> target;
2. resolves the targets (and the 100 article titles themselves) to Wikidata items;
3. asks the Wikidata SPARQL endpoint which item belongs to which class in
   ``config.WIKIDATA_CLASSES`` and fetches German labels and aliases.

The result is two CSV files (links.csv, kb.csv). ``KnowledgeBase`` loads them
and answers "what type is this name?" and "which item is this name?", which
the pipeline uses for gazetteer features, type disambiguation and for
normalising name variants (Bach / J. S. Bach / Johann Sebastian Bach -> Q1339).
"""
from __future__ import annotations

import csv
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .. import config, corpus, fetch


# --------------------------------------------------------------------------- links
_LINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|([^\[\]]+))?\]\]([a-zäöüß]*)")


def _strip_templates(wt: str) -> str:
    """Remove {{templates}} (nested), <ref>s and comments: infobox links are not running text."""
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    out, depth, i = [], 0, 0
    while i < len(wt):
        if wt.startswith("{{", i):
            depth += 1; i += 2
        elif wt.startswith("}}", i) and depth:
            depth -= 1; i += 2
        else:
            if not depth:
                out.append(wt[i])
            i += 1
    return "".join(out)


def wikitext_links(wikitext: str) -> list[tuple[str, str]]:
    """(anchor text, target title) for every article link in running text.

    ``[[Johann Sebastian Bach|Bach]]s`` -> ("Bachs", "Johann Sebastian Bach"): the
    letters glued after ``]]`` belong to the anchor, as MediaWiki renders them.
    """
    out = []
    for m in _LINK.finditer(_strip_templates(wikitext)):
        target = m.group(1).strip().replace("_", " ")
        if ":" in target:                       # Datei:, Kategorie:, Hilfe:, interwiki
            continue
        anchor = re.sub(r"\s+", " ", (m.group(2) or m.group(1)) + m.group(3)).strip()
        anchor = re.sub(r"'{2,}", "", anchor)
        out.append((anchor, target[:1].upper() + target[1:]))
    return out


def _intro_wikitext(session, revids: list[int]) -> dict[int, str]:
    """revision id -> wikitext of section 0 (the intro), 20 revisions per request."""
    out = {}
    for batch in fetch._chunks([str(r) for r in revids], 20):
        d = fetch._get(session, dict(action="query", format="json", formatversion=2, prop="revisions",
                                     rvprop="content|ids", rvslots="main", rvsection=0, revids="|".join(batch)))
        for p in d["query"]["pages"]:
            for r in p.get("revisions", []):
                out[r["revid"]] = r["slots"]["main"]["content"]
        time.sleep(config.REQUEST_PAUSE)
    return out


def _wikibase_items(session, titles: list[str]) -> dict[str, str]:
    """Article title -> Wikidata QID (following redirects), 50 titles per request."""
    out: dict[str, str] = {}
    for batch in fetch._chunks(sorted(set(titles)), 50):
        d = fetch._get(session, dict(action="query", format="json", formatversion=2, redirects=1,
                                     prop="pageprops", ppprop="wikibase_item", titles="|".join(batch)))
        q = d.get("query", {})
        resolve = {t: t for t in batch}
        for key in ("normalized", "redirects"):
            m = {x["from"]: x["to"] for x in q.get(key, [])}
            resolve = {k: m.get(v, v) for k, v in resolve.items()}
        qid = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
        for t in batch:
            if qid.get(resolve[t]):
                out[t] = qid[resolve[t]]
        time.sleep(config.REQUEST_PAUSE / 2)
    return out


def _sparql(session, query: str) -> list[dict]:
    for attempt in range(config.MAX_RETRIES):
        r = session.get(config.SPARQL_URL, params={"query": query, "format": "json"}, timeout=90)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 5 * (attempt + 1))))
            continue
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    r.raise_for_status()
    return []


def _pairs(session, query_body: str, ids: list[str], batch: int = 200) -> list[tuple[str, str]]:
    """Run ``SELECT ?a ?b`` for VALUES ?a in ids (batched) -> (a, b) QID pairs."""
    out = []
    for chunk in fetch._chunks(sorted(set(ids)), batch):
        vals = " ".join(f"wd:{q}" for q in chunk)
        for r in _sparql(session, f"SELECT ?a ?b WHERE {{ VALUES ?a {{ {vals} }} {query_body} }}"):
            out.append((r["a"]["value"].rsplit("/", 1)[1], r["b"]["value"].rsplit("/", 1)[1]))
        time.sleep(config.REQUEST_PAUSE / 2)
    return out


def _has_symbol(name: str) -> bool:
    """Wikidata aliases sometimes are emoji or pictographs; they are no names."""
    return any(unicodedata.category(c) == "So" for c in name)


def _wikidata_types(session, qids: list[str], progress: bool = False, max_depth: int = 3) -> dict[str, dict]:
    """QID -> {type, all_types, label, aliases}.

    The class hierarchy is walked level by level with cheap queries (instance of,
    then subclass of, up to ``max_depth`` levels) and resolved locally. The type
    reached on the *shortest* path wins, ties broken by the order of
    ``config.WIKIDATA_CLASSES``: the Berlin Wall is a structure after 2 steps
    (LOC) and an "administrative entity" only after 3 (GPE), so it is LOC.
    """
    target = {c: t for t, cs in config.WIKIDATA_CLASSES.items() for c in cs}
    order = list(config.WIKIDATA_CLASSES)
    instance_of: dict[str, set] = defaultdict(set)
    for a, b in _pairs(session, "?a wdt:P31 ?b .", qids):
        instance_of[a].add(b)
    # items that are themselves classes ("Bundesstaat", "Hauptstadt") are concepts, not names
    is_class = {a for a, _ in _pairs(session, "?a wdt:P279 ?b .", qids)}
    parents: dict[str, set] = defaultdict(set)
    frontier = set().union(*instance_of.values()) if instance_of else set()
    seen = set(frontier)
    for depth in range(max_depth):
        if progress:
            print(f"  class level {depth}: {len(frontier)} classes", flush=True)
        for a, b in _pairs(session, "?a wdt:P279 ?b .", list(frontier - set(target))):
            parents[a].add(b)
        frontier = set().union(*(parents[c] for c in frontier)) - seen
        seen |= frontier

    def class_types(c: str) -> dict[str, int]:
        best: dict[str, int] = {}
        level, d, visited = {c}, 0, set()
        while level and d <= max_depth:
            for x in level:
                if x in target:
                    best.setdefault(target[x], d)
            visited |= level
            level = set().union(*(parents.get(x, set()) for x in level)) - visited
            d += 1
        return best

    names: dict[str, dict] = {}
    for chunk in fetch._chunks(sorted(set(qids)), 200):
        vals = " ".join(f"wd:{q}" for q in chunk)
        for r in _sparql(session, f"""
            SELECT ?item ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
              VALUES ?item {{ {vals} }}
              OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "de") }}
              OPTIONAL {{ ?item skos:altLabel ?alias FILTER(LANG(?alias) = "de") }}
            }} GROUP BY ?item ?label"""):
            names[r["item"]["value"].rsplit("/", 1)[1]] = r
        if progress:
            print(f"  labels {len(names)}/{len(set(qids))}", flush=True)
        time.sleep(config.REQUEST_PAUSE / 2)

    info: dict[str, dict] = {}
    for q in set(qids):
        depth: dict[str, int] = {}
        for c in instance_of.get(q, ()):
            for t, d in class_types(c).items():
                depth[t] = min(depth.get(t, 99), d)
        types = sorted(depth, key=lambda t: (depth[t], order.index(t)))
        r = names.get(q, {})
        typ = types[0] if types and types[0] != "O" and q not in is_class else ""
        aliases = [x for x in r.get("aliases", {}).get("value", "").split("|") if x and not _has_symbol(x)]
        info[q] = {"type": typ, "all_types": "|".join(types),
                   "label": r.get("label", {}).get("value", ""), "aliases": "|".join(aliases)}
    return info


def build(progress: bool = True) -> tuple[int, int]:
    """Download links + Wikidata types and write links.csv and kb.csv. Needs internet."""
    config.NER_DIR.mkdir(parents=True, exist_ok=True)
    session = fetch._session()
    sources = corpus.read_sources()
    wikitext = _intro_wikitext(session, [int(s["revision_id"]) for s in sources])
    rows = []
    for s in sources:
        rows.append((s["file"], s["title"], s["title"]))          # the article's own subject
        rows += [(s["file"], a, t) for a, t in wikitext_links(wikitext.get(int(s["revision_id"]), ""))]
    if progress:
        print(f"  {len(rows)} links in {len(wikitext)} intros", flush=True)
    qid = _wikibase_items(session, [t for _, _, t in rows])
    if progress:
        print(f"  {len(qid)} of {len({t for _, _, t in rows})} link targets have a Wikidata item", flush=True)
    info = _wikidata_types(session, list(qid.values()), progress)
    for target, q in qid.items():                  # items without a German label: use the article title
        if q in info and not info[q]["label"]:
            info[q]["label"] = target
    with open(config.NER_LINKS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "anchor", "target", "qid", "type"])
        for file, anchor, target in rows:
            q = qid.get(target, "")
            w.writerow([file, anchor, target, q, info.get(q, {}).get("type", "")])
    with open(config.NER_KB, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "all_types", "label", "aliases"])
        for q, d in sorted(info.items(), key=lambda kv: int(kv[0][1:])):
            w.writerow([q, d["type"], d["all_types"], d["label"], d["aliases"]])
    return len(rows), sum(1 for d in info.values() if d["type"])


# --------------------------------------------------------------------------- lookup
def fold(text: str) -> str:
    """Matching key that ignores case, umlaut spelling and diacritics.

    'München', 'MÜNCHEN', 'Muenchen' and 'muenchen' -> 'muenchen'. Umlauts become
    ae/oe/ue (the German transliteration), not plain vowels, so 'Händel' and
    'Handel' (trade) stay different. Other diacritics are dropped: 'Zürich' is
    handled as an umlaut, 'Café' -> 'cafe'.
    """
    t = text.lower().replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\s\-‐–]+", " ", t).strip()


ARTICLES = {"der", "die", "das", "des", "dem", "den", "the", "le", "la", "les", "il"}
_ENDINGS = ("es", "en", "er", "em", "e", "s", "n")


def stem_key(text: str) -> str:
    """``fold`` plus removal of case/adjective endings per word, for inflected names:
    'Europäischen Union' and 'Europäische Union' -> 'europaisch union'."""
    out = []
    for w in fold(text).split():
        for e in _ENDINGS:
            if len(w) > 4 and w.endswith(e):
                w = w[:-len(e)]
                break
        out.append(w)
    return " ".join(out)


def strip_genitive(name: str) -> str:
    """'Deutschlands' -> 'Deutschland', 'Bachs' -> 'Bach' (not 'Paris', 'Mainz')."""
    if len(name) > 4 and name.endswith("s") and not name.endswith(("ss", "us", "is", "os", "as", "x")):
        return name[:-1]
    return name


@dataclass
class Entry:
    qid: str
    type: str
    label: str
    names: set = field(default_factory=set)


class KnowledgeBase:
    """Name -> Wikidata item lookup with variant handling (see ``fold``)."""

    def __init__(self, entries: dict[str, Entry], anchors: Counter):
        self.entries = entries
        self.anchors = anchors                   # (folded name, qid) -> how often linked
        self.index: dict[str, set[str]] = defaultdict(set)
        self.stem_index: dict[str, set[str]] = defaultdict(set)      # fallback for inflected multi-word names
        for e in entries.values():
            for n in e.names:
                self.index[fold(n)].add(e.qid)
                if len(n.split()) > 1:
                    self.stem_index[stem_key(n)].add(e.qid)
        self.first_tokens = {k.split()[0] for k in self.index} | {k.split()[0] for k in self.stem_index}
        self.max_len = max((len(k.split()) for k in self.index), default=1)

    @classmethod
    def load(cls, files: set[str] | None = None) -> "KnowledgeBase":
        """Load the KB. With ``files``, keep only items linked from those texts, so an
        evaluation on held-out texts never sees names that only their own links provide."""
        entries: dict[str, Entry] = {}
        allowed: set[str] | None = None
        if files is not None and config.NER_LINKS.exists():
            with open(config.NER_LINKS, encoding="utf-8") as f:
                allowed = {r["qid"] for r in csv.DictReader(f) if r["file"] in files}
        if config.NER_KB.exists():
            with open(config.NER_KB, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if not r["type"] or (allowed is not None and r["qid"] not in allowed):
                        continue
                    names = {r["label"]} | {a for a in r["aliases"].split("|") if a}
                    # "Der Deutsche Bundestag": an article is not part of the name
                    names = {n for n in names if len(n) > 1 and n.split()[0].lower() not in ARTICLES}
                    entries[r["qid"]] = Entry(r["qid"], r["type"], r["label"], names)
        anchors: Counter = Counter()
        if config.NER_LINKS.exists():
            with open(config.NER_LINKS, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if files is not None and r["file"] not in files:
                        continue
                    if r["qid"] in entries and _is_name_anchor(r["anchor"], entries[r["qid"]]):
                        entries[r["qid"]].names.add(r["anchor"])
                        anchors[(fold(r["anchor"]), r["qid"])] += 1
        return cls(entries, anchors)

    def candidates(self, name: str) -> list[Entry]:
        """All items this name can refer to, most frequently linked first."""
        keys = {fold(name), fold(strip_genitive(name))}
        if name.endswith("s") and len(name) > 4:          # genitive of names in -a: "Europas"
            keys.add(fold(name[:-1]))
        qids = set().union(*(self.index.get(k, set()) for k in keys))
        if not qids and " " in name.strip():
            qids = set(self.stem_index.get(stem_key(name), set()))
        if not qids and len(name) > 5 and name.endswith("n") and " " not in name:   # dative plural: Niederlanden
            qids = set(self.index.get(fold(name[:-1]), set()))
        return sorted((self.entries[q] for q in qids),
                      key=lambda e: -sum(self.anchors[(k, e.qid)] for k in keys))

    def types(self, name: str) -> set[str]:
        return {e.type for e in self.candidates(name)}

    def longest_match(self, words: list[str], start: int) -> int:
        """Length of the longest KB name starting at words[start] (0 = none)."""
        w = words[start]
        if not {fold(w), fold(strip_genitive(w)), stem_key(w)} & self.first_tokens:
            return 0
        for n in range(min(self.max_len, len(words) - start), 0, -1):
            if self.candidates(" ".join(words[start:start + n])):
                return n
        return 0


def _is_name_anchor(anchor: str, e: Entry) -> bool:
    """Keep anchors that are a *name* of the item, not a description of it.

    'Johann Sebastian Bach' / 'Bach' (persons only) / 'Deutschlands' /
    'Europäischen Union' -> the item: yes
    'Bundeshauptstadt' -> Berlin, 'deutschen' -> Deutschland, 'Mitte' -> Berlin-Mitte,
    'Hamburger' -> Hamburg (a derived adjective, not the name),
    'Finanzplatz' -> Finanzplatz Frankfurt am Main: no
    """
    if not anchor[:1].isupper() or len(anchor) < 2:
        return False
    a, sa = fold(strip_genitive(anchor)), stem_key(anchor)
    for n in e.names:
        f = fold(n)
        if a == f or (" " in a and sa == stem_key(n)) or (e.type == "PER" and a == f.split()[-1]):
            return True
    return False
