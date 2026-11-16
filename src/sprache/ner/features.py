"""Token representation and CRF features.

A token is the tuple ``(word, tag, gaz, ent)``:
    word  the surface form from the NLTK tokenizer
    tag   STTS part-of-speech tag from spaCy, run on the NLTK tokens (spaCy
          complements NLTK here: NLTK ships no German POS model; ``NE`` =
          proper noun is the single most useful NER feature for German,
          where capitalisation marks every noun and so says little)
    gaz   BIO label of the longest knowledge-base name covering the token
    ent   spaCy's own NER label (BIO, PER/ORG/LOC/MISC): a second opinion that
          covers names the knowledge base does not know (stacking)

The feature function only uses information that generalises to unseen names:
word shape, affixes, POS, gazetteer type, cue words in a +-2 window. That is
what makes a model trained on a few thousand automatically labelled tokens
usable (the low-resource setting).
"""
from __future__ import annotations

import re

from .kb import KnowledgeBase

Token = tuple[str, str, str, str]

PERSON_CUES = {"herr", "frau", "dr.", "prof.", "präsident", "präsidentin", "kanzler", "kanzlerin", "könig",
               "königin", "kaiser", "kaiserin", "papst", "fürst", "graf", "herzog", "sankt", "st.", "minister",
               "ministerin", "bürgermeister", "komponist", "schriftsteller", "maler", "physiker", "von", "van", "de"}
ORG_SUFFIXES = {"gmbh", "ag", "se", "kg", "e.", "v.", "ev", "e.v.", "inc.", "ltd.", "co.", "kgaa", "mbh", "ug"}
ORG_HEADS = {"universität", "hochschule", "verein", "verband", "partei", "bank", "gesellschaft", "konzern",
             "gruppe", "institut", "akademie", "museum", "theater", "orchester", "club", "klub", "stiftung",
             "ministerium", "bundesamt", "amt", "gericht", "rat", "kommission", "werke", "verlag",
             "stadtverwaltung", "regierung", "landtag", "bundestag", "senat", "parlament", "referat"}
LOC_HEADS = {"fluss", "see", "berg", "gebirge", "insel", "tal", "meer", "kanal", "wald", "platz", "straße",
             "bucht", "halbinsel", "küste", "region", "landschaft", "metropolregion", "agglomeration",
             "planungsregion", "flughafen", "bahnhof", "hauptbahnhof", "schloss", "nationalpark", "naturpark"}
# compound endings that decide the type of a name: "Staatsregierung", "Hoftheater", "Theresienwiese"
ORG_HEAD_SUFFIXES = ("regierung", "theater", "rat", "tag", "ministerium", "gericht", "orchester", "museum",
                     "verwaltung", "universität", "hochschule", "partei", "verband", "verein", "gesellschaft",
                     "bank", "akademie", "institut", "referat", "behörde", "amt", "kommission", "versammlung",
                     "parlament", "senat", "konzern", "stiftung", "werke", "verlag", "chor", "oper")
LOC_HEAD_SUFFIXES = ("region", "agglomeration", "flughafen", "bahnhof", "gebirge", "wiese", "park", "platz",
                     "straße", "kirche", "dom", "brücke", "schloss", "turm", "halle", "fluss", "insel")


def head_class(word: str) -> str:
    """ORG / LOC / GPE if the noun names a kind of organisation or place, else ''."""
    lw = word.lower()
    forms = {lw, lw[:-1] if lw.endswith("s") else lw, lw[:-2] if lw.endswith("es") else lw}
    if forms & ORG_HEADS or any(f.endswith(ORG_HEAD_SUFFIXES) for f in forms):
        return "ORG"
    if forms & LOC_HEADS or any(f.endswith(LOC_HEAD_SUFFIXES) for f in forms):
        return "LOC"
    if forms & GPE_HEADS:
        return "GPE"
    return ""
GPE_HEADS = {"stadt", "gemeinde", "land", "staat", "kreis", "landkreis", "bezirk", "provinz", "kanton",
             "bundesland", "republik", "königreich", "hauptstadt", "landeshauptstadt", "dorf", "ort"}
SPATIAL_PREPS = {"in", "im", "nach", "aus", "bei", "von", "vom", "zu", "zur", "zum", "nahe", "östlich",
                 "westlich", "nördlich", "südlich", "über", "an", "am", "durch"}


def shape(w: str) -> str:
    s = re.sub(r"[A-ZÄÖÜ]", "X", w)
    s = re.sub(r"[a-zäöüß]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)          # Xxxxxx -> Xxx


def gazetteer_tags(words: list[str], kb: KnowledgeBase) -> list[str]:
    """Greedy longest-match KB lookup over capitalised tokens -> BIO tags."""
    tags = ["O"] * len(words)
    i = 0
    while i < len(words):
        n = kb.longest_match(words, i) if words[i][:1].isupper() else 0
        if n:
            typ = kb.candidates(" ".join(words[i:i + n]))[0].type
            tags[i] = f"B-{typ}"
            for k in range(i + 1, i + n):
                tags[k] = f"I-{typ}"
            i += n
        else:
            i += 1
    return tags


def _word_feats(tok: Token, prefix: str) -> dict[str, str]:
    w, tag, gaz, ent = tok
    lw = w.lower()
    return {f"{prefix}lower": lw, f"{prefix}tag": tag, f"{prefix}gaz": gaz, f"{prefix}ent": ent,
            f"{prefix}shape": shape(w),
            f"{prefix}title": str(w[:1].isupper()), f"{prefix}cue": (
                "per" if lw in PERSON_CUES else "orgsuf" if lw in ORG_SUFFIXES else "orghead" if lw in ORG_HEADS
                else "lochead" if lw in LOC_HEADS else "gpehead" if lw in GPE_HEADS
                else "prep" if lw in SPATIAL_PREPS else "-")}


def token_features(tokens: list[Token], i: int) -> dict[str, str | float]:
    """Feature dict for tokens[i] (signature required by nltk.tag.CRFTagger)."""
    w, tag, gaz, ent = tokens[i]
    lw = w.lower()
    f: dict[str, str | float] = {
        "bias": 1.0, "word": lw, "suf3": lw[-3:], "suf4": lw[-4:], "pre3": lw[:3], "shape": shape(w),
        "tag": tag, "gaz": gaz, "gaztype": gaz[2:] or "O", "ent": ent, "ent+gaz": f"{ent[2:]}|{gaz[2:]}",
        "title": str(w[:1].isupper()),
        "upper": str(w.isupper() and len(w) > 1), "hyphen": str("-" in w[1:]), "digit": str(any(c.isdigit() for c in w)),
        "first": str(i == 0), "len": str(min(len(w), 12) // 4), "tag+title": f"{tag}|{w[:1].isupper()}",
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            for k, v in _word_feats(tokens[j], f"{off}:").items():
                f[k] = v
        else:
            f[f"{off}:pad"] = "BOS" if j < 0 else "EOS"
    if i > 0:
        f["-1:tag|tag"] = f"{tokens[i - 1][1]}|{tag}"
        f["-1:gaz|gaz"] = f"{tokens[i - 1][2]}|{gaz}"
    return f
