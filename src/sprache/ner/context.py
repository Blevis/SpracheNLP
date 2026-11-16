"""Context layer: type ambiguity, metonymy, nested entities and name variation.

The CRF proposes spans and types from local features. This layer revisits each
span with evidence the CRF does not see well:

Type ambiguity  "Bosch" (person or company?), "Paris" (city or person?).
                Votes from the CRF, the knowledge base, cue words around the
                span (``Herr``, ``GmbH``, ``Universität``), then one type per
                name per document (a name keeps its reading within one text).
Metonymy        "Berlin lehnt den Vorschlag ab": the *place* stands for the
                government. A GPE/LOC that is the grammatical subject (spaCy
                dependency label ``sb``) of a verb that needs a human or
                institutional agent gets the contextual type ORG; its literal
                type is kept as ``base_type`` (reading "place-for-institution").
Nesting         "Universität Wien" (ORG) contains "Wien" (GPE): KB names inside a
                multi-token span become the inner BIO layer.
Variation       "Bachs", "Bach", "J. S. Bach", "Johann Sebastian Bach" all get one
                canonical name (and Wikidata id when known): genitive stripping,
                umlaut-insensitive KB lookup, and surname -> full name within a text.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .. import config
from .doc import Sentence
from .features import ORG_SUFFIXES, PERSON_CUES, head_class
from .kb import KnowledgeBase, fold, strip_genitive

PERSON_TITLES = PERSON_CUES - {"von", "van", "de", "sankt", "st."}

# Verbs whose subject must be a person or an institution (lemmas as spaCy gives them).
AGENTIVE_VERBS = {
    "beschließen", "erklären", "ankündigen", "verhandeln", "fordern", "kritisieren", "ablehnen", "zustimmen",
    "unterzeichnen", "verbieten", "erlauben", "reagieren", "protestieren", "drohen", "warnen", "verurteilen",
    "entscheiden", "mitteilen", "bestätigen", "dementieren", "anerkennen", "boykottieren", "sanktionieren",
    "ratifizieren", "beantragen", "verlangen", "bestreiten", "betonen", "begrüßen", "unterstützen",
    "einladen", "ausweisen", "zahlen", "investieren", "kaufen", "verkaufen", "liefern", "exportieren",
    "importieren", "planen", "wählen", "regieren", "antworten", "sagen", "melden",
}
# Verbs whose subject, for a place name, is a sports team (place-for-team).
SPORT_VERBS = {"gewinnen", "besiegen", "schlagen", "verlieren", "unterliegen", "siegen"}   # not "führen", "spielen": "führt durch", "spielt eine Rolle"
# Verbs whose object, for a company name with an article, is a product (org-for-product):
# "Er konstruierte den ersten Volkswagen", "sie fährt einen Porsche".
PRODUCT_VERBS = {"fahren", "kaufen", "bauen", "konstruieren", "besitzen", "verkaufen", "entwickeln",
                 "produzieren", "steuern", "reparieren", "fliegen", "bestellen"}
DETERMINERS = {"ein", "einen", "einem", "eines", "eine", "den", "dem", "der", "des", "die", "das", "seinen",
               "ihren", "seinem", "ihrem", "neuen", "ersten", "alten"}
PLACE_TYPES = {"GPE", "LOC"}
STRONG = 3.0


@dataclass
class Mention:
    file: str
    sent: int
    start: int
    end: int
    text: str
    type: str                 # contextual type (what the BIO label says)
    base_type: str = ""       # literal/KB type before metonymy resolution
    reading: str = "literal"  # literal | place-for-institution | place-for-team
    layer: str = "outer"
    canonical: str = ""
    qid: str = ""
    score: float = 0.0        # winning vote of the type decision


# ------------------------------------------------------------------ type ambiguity
def type_votes(s: Sentence, start: int, end: int, crf_type: str, kb: KnowledgeBase) -> Counter:
    """Votes per type: CRF 2, knowledge base 2.5 (+0.5 for runner-up readings), strong
    cue words 3: a title before the name, a legal form after it, or a head noun that
    names a kind of organisation or place ("Universität ...", "... Staatsregierung")."""
    v: Counter = Counter()
    if crf_type:
        v[crf_type] += 2.0
    cands = kb.candidates(s.text(start, end))
    for k, c in enumerate(cands[:3]):
        v[c.type] += 2.5 if k == 0 else 0.5
    prev = s.words[start - 1].lower() if start else ""
    nxt = s.words[end].lower() if end < len(s.words) else ""
    last = s.words[end - 1].lower()
    if prev in PERSON_TITLES:
        v["PER"] += STRONG
    if prev == "bei" and not {c.type for c in cands} & PLACE_TYPES:
        v["ORG"] += 1.5                                # "arbeitet bei Bosch": employer, not place
    if nxt in ORG_SUFFIXES or last in ORG_SUFFIXES:
        v["ORG"] += STRONG
    if end - start > 1:
        h = head_class(s.words[start]) or head_class(s.words[end - 1])
    else:
        h = head_class(s.words[start]) if not cands else ""
        h = "" if h == "GPE" else h                    # "Stadt" alone is no name
    if h:
        v[h] += STRONG
    return v


# ------------------------------------------------------------------ plausibility and span repair
NOUNISH = {"NE", "NN", "FM", "XY", "TRUNC", "CARD", "NNE"}
NOBLE_TITLES = {"herzog", "herzogin", "könig", "königin", "kaiser", "kaiserin", "fürst", "fürstin", "graf",
                "gräfin", "kurfürst", "markgraf", "landgraf", "erzherzog", "prinz", "prinzessin", "großherzog"}
DESCRIPTORS = {"stadt", "gemeinde", "land", "dorf", "hauptstadt", "landeshauptstadt", "bundeshauptstadt"}


def plausible(s: Sentence, start: int, end: int, kb: KnowledgeBase) -> bool:
    """Reject spans that cannot be names: no noun in them ("Amtlich", "Weimarer"), or a
    single common noun the KB does not know and spaCy does not call an entity ("Stadt")."""
    if not s.tags:
        return True
    tags = s.tags[start:end]
    if not set(tags) & NOUNISH:
        return False
    if end - start == 1 and tags[0] == "NN" and not kb.candidates(s.words[start]) \
            and (not s.ents or s.ents[start] == "O"):
        return False
    return True


def trim_descriptor(s: Sentence, start: int, end: int) -> int:
    """New start: 'Stadt Frankfurt am Main' -> 'Frankfurt am Main' (the noun only describes)."""
    if end - start > 1 and s.words[start].lower() in DESCRIPTORS:
        return start + 1
    return start


def split_title(s: Sentence, start: int, end: int, kb: KnowledgeBase) -> list[tuple[int, int, str]] | None:
    """'Carl August von Sachsen-Weimar-Eisenach' -> PER 'Carl August' + GPE 'Sachsen-Weimar-Eisenach'
    when the part after von/zu is a known place (not for 'Otto von Bismarck')."""
    noble = start and s.words[start - 1].lower() in NOBLE_TITLES      # "Herzog Carl August von X"
    for j in range(start + 1, end - 1):
        if s.words[j] in ("von", "zu"):
            place = [c for c in kb.candidates(s.text(j + 1, end)) if c.type in PLACE_TYPES]
            if place or noble:
                return [(start, j, "PER"), (j + 1, end, place[0].type if place else "GPE")]
    return None


def coordination_consistency(s: Sentence, mentions: list["Mention"], kb: KnowledgeBase) -> None:
    """Names in one list share a type: in "Aare , Neckar , Main , Mosel und Maas" the
    members the KB knows as rivers (LOC) decide for the unknown ones."""
    links = {",", "und", "oder", "sowie", "bzw.", "/", "&"}
    ms = sorted((m for m in mentions if m.type in config.ENAMEX and m.layer == "outer"), key=lambda m: m.start)
    groups, cur = [], []
    for m in ms:
        if cur and all(w in links for w in s.words[cur[-1].end:m.start]) and m.start > cur[-1].end:
            cur.append(m)
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [m]
    if len(cur) > 1:
        groups.append(cur)
    for g in groups:
        known = Counter(kb.candidates(m.text)[0].type for m in g if kb.candidates(m.text))
        if not known:
            continue
        t, n = known.most_common(1)[0]
        if n < 2 and len(known) > 1:
            continue
        for m in g:
            if not kb.candidates(m.text) and m.score < STRONG + 2 and m.base_type != "PER":
                m.base_type = t
                if m.reading == "literal":
                    m.type = t


def one_sense_per_document(mentions: list[Mention]) -> None:
    """Give every mention of a name the document-majority type unless it has strong local evidence."""
    by_name: dict[tuple, list[Mention]] = defaultdict(list)
    for m in mentions:
        if m.type in config.ENAMEX and m.layer == "outer":
            by_name[(m.file, fold(strip_genitive(m.text)))].append(m)
    # a surname alone is the person named in full elsewhere in the text: "Bachs Werke"
    surnames = {(m.file, fold(m.text.split()[-1])) for m in mentions
                if m.base_type == "PER" and len(m.text.split()) > 1}
    for m in mentions:
        if (m.layer == "outer" and len(m.text.split()) == 1 and m.base_type != "PER"
                and (m.file, fold(strip_genitive(m.text))) in surnames and m.score < STRONG + 2):
            m.base_type = m.type = "PER"
            m.reading = "literal"
    for ms in by_name.values():
        if len(ms) < 2:
            continue
        votes: Counter = Counter()
        for m in ms:
            votes[m.base_type] += m.score
        best = votes.most_common(1)[0][0]
        for m in ms:
            if m.base_type != best and m.score < STRONG + 2:
                m.base_type = best
                if m.reading == "literal":
                    m.type = best


# ------------------------------------------------------------------ metonymy
def _verb_lemma(s: Sentence, v: int) -> str:
    """Lemma of the full verb at v: separable particle re-attached ("lehnt ... ab" ->
    "ablehnen"), and modal/auxiliary replaced by its verb ("hat ... beschlossen")."""
    for k in range(len(s.words)):
        if s.heads[k] == v and s.deps[k] == "oc" and s.tags[k].startswith("V") and k != v:
            return _verb_lemma(s, k)
    particles = [s.words[k].lower() for k in range(len(s.words)) if s.heads[k] == v and s.deps[k] == "svp"]
    return (particles[0] if particles else "") + s.lemmas[v].lower()


def _role(s: Sentence, i: int) -> tuple[str, str] | None:
    """(role, verb lemma) of token i: role is 'subject', 'object' or 'against'
    ("gegen Brasilien"). Coordination is followed: in "Berlin und Paris
    verhandeln", Paris is a conjunct of Berlin, which is the subject."""
    if not s.deps:
        return None
    seen = set()
    while s.deps[i] in ("cj", "cd") and i not in seen and s.heads[i] != i:
        seen.add(i)
        i = s.heads[i]
    h = s.heads[i]
    if s.deps[i] == "sb":
        return "subject", _verb_lemma(s, h)
    if s.deps[i] in ("oa", "da"):
        return "object", _verb_lemma(s, h)
    if s.deps[i] == "nk" and s.words[h].lower() == "gegen" and s.heads[h] != h:
        return "against", _verb_lemma(s, s.heads[h])
    return None


def metonymy(s: Sentence, start: int, end: int, typ: str) -> tuple[str, str]:
    """(contextual type, reading) for a place name in this sentence.

    place-for-institution: subject of an agentive verb ("Berlin lehnt ... ab")
    place-for-team:        subject, object or "gegen"-opponent of a sports verb
                           ("Deutschland gewann gegen Brasilien")
    org-for-product:       a company name with an article as object of a verb like
                           fahren/bauen ("konstruierte den ersten Volkswagen") -> type O
    """
    if typ == "ORG" and s.deps and start and s.words[start - 1].lower() in DETERMINERS:
        head = next((i for i in range(start, end) if not start <= s.heads[i] < end or s.heads[i] == i), end - 1)
        r = _role(s, head)
        if r and r[0] == "object" and r[1] in PRODUCT_VERBS:
            return "O", "org-for-product"
    if typ not in PLACE_TYPES or not s.deps:
        return typ, "literal"
    # syntactic head of the span = the token whose head lies outside the span
    head = next((i for i in range(start, end) if not start <= s.heads[i] < end or s.heads[i] == i), end - 1)
    r = _role(s, head)
    if not r:
        return typ, "literal"
    role, lemma = r
    if lemma in SPORT_VERBS:
        return "ORG", "place-for-team"
    if role == "subject" and lemma in AGENTIVE_VERBS:
        return "ORG", "place-for-institution"
    return typ, "literal"


# ------------------------------------------------------------------ head nouns -> nested names
# "Landkreis [München]", "Universität [Wien]", "Metropolregion [Berlin]": a place
# name after one of these nouns is the *inner* entity of a bigger name whose type
# the noun decides. For territorial titles ("Kanton Graubünden", "Freistaat
# Bayern") the bigger name is the same entity, so no inner span is kept.
HEAD_EXTENSION = {
    "GPE": {"landkreis", "kreis", "regierungsbezirk", "erzbistum", "bistum"},
    "GPE=": {"kanton", "freistaat", "herzogtum", "fürstentum", "königreich", "kaiserreich", "grafschaft",
             "kurfürstentum", "mark", "provinz", "bundesland", "republik", "markgrafschaft"},
    "LOC": {"metropolregion", "agglomeration", "planungsregion", "flughafen", "hauptbahnhof", "bahnhof",
            "schloss", "nationalpark", "naturpark", "hafen", "messe"},
    "ORG": {"universität", "hochschule", "stadtverwaltung", "verwaltung", "landtag", "stadtrat", "bundesamt",
            "landesamt", "landesmuseum", "stadtwerke", "landesbank", "oper", "philharmonie", "staatsoper"},
}


def head_extension(words: list[str], start: int, typ: str) -> tuple[str, bool] | None:
    """(new type, keep inner?) if words[start-1] is a head noun that forms a bigger name."""
    if not start or typ not in PLACE_TYPES:
        return None
    w = words[start - 1]
    if not w[:1].isupper():
        return None
    lw = w.lower()
    for key, heads in HEAD_EXTENSION.items():
        if lw in heads or (lw.endswith("es") and lw[:-2] in heads) or (lw.endswith("s") and lw[:-1] in heads):
            return key.rstrip("="), not key.endswith("=")
    return None


def merge_rules(enamex: list[tuple[int, int, str]], numtime: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Combine ENAMEX and NUMEX/TIMEX spans. On overlap the ENAMEX span wins, unless the
    rule span strictly contains it ("[DATE Mitte September]" over a GPE "Mitte")."""
    out = list(enamex)
    for r in numtime:
        inside = [e for e in out if r[0] <= e[0] and e[1] <= r[1] and (e[0], e[1]) != (r[0], r[1])]
        overlap = [e for e in out if e[0] < r[1] and r[0] < e[1]]
        if overlap and len(inside) == len(overlap):
            out = [e for e in out if e not in inside] + [r]
        elif not overlap:
            out.append(r)
    return sorted(out)


# ------------------------------------------------------------------ nesting
_ADJ_ENDINGS = ("e", "en", "er", "es", "em")


def is_nested_name(words: list[str], start: int, end: int, a: int, b: int, kb: KnowledgeBase,
                   tags: list[str] | None = None) -> bool:
    """Is words[a:b] a *different* name embedded in the name words[start:end]?

    yes: Universität [Heidelberg], Frankfurt am [Main], Büro der Vereinten Nationen in [Wien]
    no:  [Deutsche Bahn] AG, [Regierungsbezirk] Stuttgart  (the name's own beginning)
         Europäische [Parlament], Großen [Tiergartens]   (adjective + the name's own head)
         Bundesrepublik [Deutschland]                     (same Wikidata item)
    """
    if (a, b) == (start, end) or a == start or not words[a][:1].isupper():
        return False
    if b == end:
        if tags:
            adjectival = all(t == "ADJA" for t in tags[start:a])
        else:
            adjectival = all(w[:1].isupper() and w.lower().endswith(_ADJ_ENDINGS) for w in words[start:a])
        if adjectival:
            return False
    outer_q = {c.qid for c in kb.candidates(" ".join(words[start:end]))}
    inner_q = {c.qid for c in kb.candidates(" ".join(words[a:b]))}
    return bool(inner_q) and not inner_q & outer_q


def inner_spans(s: Sentence, start: int, end: int, kb: KnowledgeBase) -> list[tuple[int, int, str]]:
    """KB names strictly inside a multi-token span ("Universität [Wien]")."""
    out = []
    a = start + 1
    while a < end:
        found = 0
        for b in range(end, a, -1):
            if is_nested_name(s.words, start, end, a, b, kb, s.tags or None):
                out.append((a, b, kb.candidates(s.text(a, b))[0].type))
                found = b - a
                break
        a += found or 1
    return out


# ------------------------------------------------------------------ variation
def canonicalise(mentions: list[Mention], kb: KnowledgeBase) -> None:
    """Fill canonical name and Wikidata id; link surnames to the full name in the same text."""
    full_names: dict[tuple[str, str], Mention] = {}
    for m in mentions:
        if m.type in config.ENAMEX:
            cands = [c for c in kb.candidates(m.text) if c.type == m.base_type] or kb.candidates(m.text)
            if cands:
                m.canonical, m.qid = cands[0].label, cands[0].qid
            else:
                m.canonical = strip_genitive(m.text)
            if m.base_type == "PER" and len(m.text.split()) > 1:
                full_names.setdefault((m.file, fold(strip_genitive(m.text.split()[-1]))), m)
        else:
            m.canonical = m.text
    for m in mentions:
        if m.base_type == "PER" and len(m.text.split()) == 1:
            full = full_names.get((m.file, fold(strip_genitive(m.text))))
            if full:
                m.canonical, m.qid = full.canonical, full.qid or m.qid
