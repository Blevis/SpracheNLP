"""Per-sentence analysis helpers: token tables, displaCy export, probes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Doc

from . import config

TOKEN_COLUMNS = ["i", "token", "lemma", "pos", "tag", "morph", "dep", "head", "ent"]


def load_model(name: str = config.SPACY_MODEL) -> Language:
    try:
        return spacy.load(name)
    except OSError as e:
        raise OSError(f"spaCy model '{name}' is not installed. Run:\n"
                      f"    python -m spacy download {name}") from e


def token_table(doc: Doc, morph: bool = False) -> pd.DataFrame:
    rows = [(t.i, t.text, t.lemma_, t.pos_, t.tag_, str(t.morph), t.dep_, t.head.text,
             f"{t.ent_iob_}-{t.ent_type_}" if t.ent_type_ else "") for t in doc]
    df = pd.DataFrame(rows, columns=TOKEN_COLUMNS).set_index("i")
    return df if morph else df.drop(columns="morph")


def entity_table(doc: Doc) -> pd.DataFrame:
    return pd.DataFrame([(e.text, e.label_, e.start, e.end) for e in doc.ents],
                        columns=["entity", "label", "start", "end"])


def save_displacy(doc: Doc, stem: str, out_dir: Path = config.FIGURES) -> dict[str, Path]:
    """Write <stem>_dep.svg and <stem>_ent.html; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dep = out_dir / f"{stem}_dep.svg"
    ent = out_dir / f"{stem}_ent.html"
    dep.write_text(displacy.render(doc, style="dep", jupyter=False,
                                   options={"distance": 95, "compact": False}), encoding="utf-8")
    ent.write_text(displacy.render(doc, style="ent", jupyter=False, page=True), encoding="utf-8")
    return {"dep": dep, "ent": ent}


def locate(sentence: str, texts: dict[str, str]) -> str | None:
    """File that contains the sentence verbatim (None if it is not in the corpus)."""
    return next((name for name, t in texts.items() if sentence in t), None)


def probe_table(nlp: Language, probes: list[str] = config.GERMAN_PROBES) -> pd.DataFrame:
    rows = []
    for p in probes:
        d = nlp(p)
        rows.append(dict(input=p, n_tokens=len(d),
                         tokens=" | ".join(t.text for t in d),
                         lemmas=" | ".join(t.lemma_ for t in d),
                         pos=" | ".join(t.pos_ for t in d)))
    return pd.DataFrame(rows)


def longest_words(docs: dict[str, Doc], n: int = 10) -> pd.DataFrame:
    seen: dict[str, tuple[str, str]] = {}
    for d in docs.values():
        for t in d:
            if t.is_alpha and t.text not in seen:
                seen[t.text] = (t.lemma_, t.pos_)
    top = sorted(seen, key=len, reverse=True)[:n]
    return pd.DataFrame([(w, len(w), *seen[w]) for w in top],
                        columns=["word", "length", "lemma", "pos"])


def contraction_lemmas(docs: dict[str, Doc],
                       forms=("am", "im", "zum", "zur", "vom", "beim", "ins", "ans")) -> pd.DataFrame:
    """How often each preposition+article contraction occurs and how it is lemmatised."""
    from collections import Counter, defaultdict
    counts, lemmas = Counter(), defaultdict(Counter)
    for d in docs.values():
        for t in d:
            if t.lower_ in forms:
                counts[t.lower_] += 1
                lemmas[t.lower_][f"{t.lemma_}/{t.tag_}"] += 1
    return pd.DataFrame([(f, counts[f], dict(lemmas[f].most_common(3))) for f in forms if counts[f]],
                        columns=["form", "count", "lemma/tag (top 3)"])
