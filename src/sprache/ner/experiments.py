"""Training, evaluation and the corpus entity index.

Data split (by text, never by sentence, so no article is on both sides):
    gold   text001-text008: one text per topic, hand-checked labels (gold.conll)
    train  the other 92 texts: silver labels from distant supervision

During evaluation the knowledge base is restricted to items linked from the
*training* texts, so names that only the gold texts' own links would supply
are unknown to the system, as they would be on new text.
"""
from __future__ import annotations

import random
from functools import lru_cache

import pandas as pd

from .. import config, corpus
from . import bio, evaluate, noise, rules, silver
from .doc import Sentence, parse, read_conll, write_conll
from .kb import KnowledgeBase
from .pipeline import NERPipeline, mentions_table, spacy_baseline
from .tagger import EntityTagger, enamex_only

GOLD_FILES = [f"text{i:03d}.txt" for i in range(1, 9)]


def all_files() -> list[str]:
    return sorted(corpus.load(config.CORPUS_CLEAN))


def train_files() -> list[str]:
    return [f for f in all_files() if f not in GOLD_FILES]


@lru_cache(maxsize=4)
def _kb(restricted: bool) -> KnowledgeBase:
    return KnowledgeBase.load(set(train_files()) if restricted else None)


def silver_corpus(nlp, restricted: bool = True) -> list[Sentence]:
    """Silver BIO for all texts: links + spaCy complement (KB restricted to training links)."""
    return silver.annotate_corpus(_kb(restricted), nlp=nlp)


def gold() -> list[Sentence]:
    return read_conll(config.NER_GOLD)


def train_tagger(nlp, sentences: list[Sentence], kb: KnowledgeBase, use_gazetteer: bool = True,
                 use_spacy: bool = True, name: str = "enamex") -> EntityTagger:
    if not sentences[0].tags:
        parse(nlp, sentences)
    t = EntityTagger(kb, use_gazetteer=use_gazetteer, use_spacy=use_spacy)
    return t.train(sentences, config.NER_MODELS / f"{name}.crf.tagger")


def _fresh(sentences: list[Sentence]) -> list[Sentence]:
    return [Sentence(s.file, list(s.words)) for s in sentences]


def _normaliser(kb: KnowledgeBase, files: list[str]) -> noise.Normaliser:
    texts = corpus.load(config.CORPUS_CLEAN)
    from . import text
    return noise.Normaliser([w for f in files for w in text.tokenize(texts[f])], kb)


def compare_systems(nlp) -> tuple[pd.DataFrame, dict]:
    """Micro F1 of every system on the gold texts, plus the per-type table of the full pipeline."""
    kb = _kb(True)
    train = [s for s in silver_corpus(nlp, True) if s.file not in GOLD_FILES]
    g = gold()
    gold_tags = [s.outer for s in g]
    results, tables = {}, {}

    # 1 spaCy alone (PER/ORG/LOC; no GPE, no NUMEX/TIMEX)
    sp = spacy_baseline(nlp, g)
    results["spaCy de_core_news_md (as is)"] = sp
    # 2 spaCy + KB for GPE/LOC + our NUMEX/TIMEX rules
    spk = spacy_baseline(nlp, g, kb)
    results["spaCy + KB types + rules"] = [_add_rules(t, s.words) for t, s in zip(spk, g)]
    # 3 dictionary baseline: KB longest match + rules, no learning
    local = {}
    results["Gazetteer + rules (no learning)"] = [silver.annotate_sentence(Sentence(s.file, list(s.words)), local, kb).outer
                                                  for s in g]
    # 4-6 CRF with growing feature sets (+ rules), no context layer
    for label, gaz, sp in (("CRF, word features only + rules", False, False),
                           ("CRF + gazetteer + rules", True, False),
                           ("CRF + gazetteer + spaCy feature + rules", True, True)):
        tg = train_tagger(nlp, train, kb, use_gazetteer=gaz, use_spacy=sp, name="eval")
        p = NERPipeline(nlp, kb, tg, use_context=False).annotate(_fresh(g))
        results[label] = [s.outer for s in p]
    # 7 full pipeline
    full = NERPipeline(nlp, kb, tg).annotate(_fresh(g))
    results["Full pipeline (CRF + context layer + rules)"] = [s.outer for s in full]

    rows = []
    for name, pred in results.items():
        sc = evaluate.score(gold_tags, pred)
        rows.append((name, sc.loc["ENAMEX", "f1"], sc.loc["NUMEX", "f1"], sc.loc["TIMEX", "f1"], sc.loc["all (micro)", "f1"]))
        tables[name] = sc
    df = pd.DataFrame(rows, columns=["system", "ENAMEX F1", "NUMEX F1", "TIMEX F1", "all F1"]).set_index("system")
    tables["inner layer"] = evaluate.score([s.inner for s in g], [s.inner for s in full], config.ENAMEX)
    return df, {"tables": tables, "full": full, "gold": g, "tagger": tg, "kb": kb}


def _add_rules(tags: list[str], words: list[str]) -> list[str]:
    spans = bio.bio_to_spans(tags)
    taken = {i for a, b, _ in spans for i in range(a, b)}
    spans += [s for s in rules.spans(words) if not taken & set(range(s[0], s[1]))]
    return bio.spans_to_bio(len(words), spans)


def learning_curve(nlp, fractions=(0.1, 0.25, 0.5, 1.0), seed: int = 0) -> pd.DataFrame:
    """ENAMEX F1 on gold when training on a fraction of the silver training texts."""
    kb = _kb(True)
    train = [s for s in silver_corpus(nlp, True) if s.file not in GOLD_FILES]
    files = sorted({s.file for s in train})
    random.Random(seed).shuffle(files)
    g = gold()
    rows = []
    for fr in fractions:
        keep = set(files[:max(1, round(len(files) * fr))])
        sub = [s for s in train if s.file in keep]
        tg = train_tagger(nlp, sub, kb, name="curve")
        for ctx in (False, True):
            p = NERPipeline(nlp, kb, tg, use_context=ctx).annotate(_fresh(g))
            rows.append((fr, len(keep), sum(len(s.words) for s in sub), ctx,
                         evaluate.micro_f1([enamex_only(s.outer) for s in g], [enamex_only(s.outer) for s in p],
                                           config.ENAMEX)))
    df = pd.DataFrame(rows, columns=["fraction", "texts", "tokens", "context", "ENAMEX F1"])
    return df.pivot_table(index=["fraction", "texts", "tokens"], columns="context", values="ENAMEX F1") \
             .rename(columns={False: "CRF only", True: "CRF + context"}).reset_index()


def noise_robustness(nlp, tagger: EntityTagger, kb: KnowledgeBase) -> pd.DataFrame:
    """All-type F1 on corrupted copies of the gold texts, without and with normalisation."""
    g = gold()
    norm = _normaliser(kb, train_files())
    rows = []
    for kind in ("clean",) + noise.KINDS:
        noisy = [Sentence(s.file, list(s.words) if kind == "clean" else noise.corrupt(s.words, kind, seed=k))
                 for k, s in enumerate(g)]
        row = [kind]
        for normalise in (False, True):
            p = NERPipeline(nlp, kb, tagger, normaliser=norm).annotate([Sentence(s.file, list(s.words)) for s in noisy],
                                                                     normalise=normalise)
            row.append(evaluate.micro_f1([s.outer for s in g], [s.outer for s in p]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["text", "F1 as is", "F1 normalised"]).set_index("text")


# ------------------------------------------------------------------ challenge set
def read_challenge() -> list[tuple[str, Sentence, list[bio.Span]]]:
    """Hand-written test sentences in bracket notation, nested brackets allowed:

        metonymy  [ORG Berlin] lehnt den Vorschlag ab .
        nested    Sie studierte an der [ORG Universität [GPE Wien]] .

    First word = phenomenon, then the tokenised sentence (tokens separated by spaces).
    """
    out = []
    for line in config.NER_CHALLENGE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        phen, annotated = line.split(None, 1)
        words, spans, stack = [], [], []
        for piece in annotated.split():
            if piece.startswith("[") and piece[1:] in config.ENTITY_TYPES:
                stack.append((len(words), piece[1:]))
                continue
            closes = len(piece) - len(piece.rstrip("]")) if len(piece) > 1 else 0
            words.append(piece[:len(piece) - closes] if closes else piece)
            for _ in range(closes):
                start, t = stack.pop()
                spans.append((start, len(words), t))
        out.append((phen, Sentence("challenge", words), spans))
    return out


def challenge(nlp, tagger: EntityTagger, kb: KnowledgeBase, normaliser: noise.Normaliser | None = None) -> pd.DataFrame:
    """Per sentence: expected vs found entities (both layers), for each phenomenon."""
    items = read_challenge()
    sents = [s for _, s, _ in items]
    pipe = NERPipeline(nlp, kb, tagger, normaliser=normaliser)
    rows = []
    for (phen, s, gold_spans), in_ in zip(items, sents):
        p = pipe.annotate([in_], normalise=phen == "noisy")[0]
        found = {(m.start, m.end, m.type) for m in p.mentions}
        g = set(gold_spans)
        fmt = lambda sp: "; ".join(f"{' '.join(s.words[a:b])}/{t}" for a, b, t in sorted(sp)) or "-"
        rows.append((phen, " ".join(s.words), fmt(g), fmt(found), len(g & found), len(g), len(found - g)))
    return pd.DataFrame(rows, columns=["phenomenon", "sentence", "expected", "found", "correct", "expected_n", "spurious"])


# ------------------------------------------------------------------ production run
def build_index(nlp) -> pd.DataFrame:
    """Train on all silver texts with the full KB, tag the whole corpus, write entities.csv."""
    kb = _kb(False)
    sil = silver.annotate_corpus(kb, nlp=nlp)
    write_conll(sil, config.NER_SILVER)
    tagger = train_tagger(nlp, sil, kb, name="enamex")
    out = NERPipeline(nlp, kb, tagger, normaliser=_normaliser(kb, all_files())).annotate(_fresh(sil))
    df = mentions_table(out)
    df.to_csv(config.NER_OUTPUT, index=False, encoding="utf-8")
    return df
