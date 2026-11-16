"""Entity-level evaluation (CoNLL style: a span counts only if start, end and type all match)."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .. import config
from . import bio

GROUPS = {"ENAMEX": config.ENAMEX, "NUMEX": config.NUMEX, "TIMEX": config.TIMEX}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold: list[list[str]], pred: list[list[str]], types=config.ENTITY_TYPES) -> pd.DataFrame:
    """Per-type and grouped precision / recall / F1 for parallel lists of BIO sequences."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred, strict=True):
        gs, ps = set(bio.bio_to_spans(g)), set(bio.bio_to_spans(p))
        for s in gs & ps:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1
    rows = []
    for t in types:
        rows.append((t, *_prf(tp[t], fp[t], fn[t]), tp[t] + fn[t]))
    for name, ts in GROUPS.items():
        ts = [t for t in ts if t in types]
        if ts:
            rows.append((name, *_prf(*(sum(c[t] for t in ts) for c in (tp, fp, fn))), sum(tp[t] + fn[t] for t in ts)))
    rows.append(("all (micro)", *_prf(sum(tp[t] for t in types), sum(fp[t] for t in types),
                                      sum(fn[t] for t in types)), sum(tp[t] + fn[t] for t in types)))
    df = pd.DataFrame(rows, columns=["type", "precision", "recall", "f1", "support"]).set_index("type")
    return df.round(3)


def micro_f1(gold, pred, types=config.ENTITY_TYPES) -> float:
    return float(score(gold, pred, types).loc["all (micro)", "f1"])


def errors(gold: list[list[str]], pred: list[list[str]], words: list[list[str]]) -> pd.DataFrame:
    """Every false positive / false negative with its text, for error analysis."""
    rows = []
    for k, (g, p, w) in enumerate(zip(gold, pred, words)):
        gs, ps = set(bio.bio_to_spans(g)), set(bio.bio_to_spans(p))
        for s, e, t in sorted(gs - ps):
            match = [x for x in ps if x[0] < e and s < x[1]]
            rows.append((k, " ".join(w[s:e]), t, ", ".join(f"{' '.join(w[a:b])}/{u}" for a, b, u in match) or "-"))
        for s, e, t in sorted(ps - gs):
            if not any(x[0] < e and s < x[1] for x in gs):
                rows.append((k, "-", "-", f"{' '.join(w[s:e])}/{t}"))
    return pd.DataFrame(rows, columns=["sentence", "gold", "gold_type", "predicted"])
