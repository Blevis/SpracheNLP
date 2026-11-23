"""Evaluation of the search engine on hand-judged queries (data/search/queries.tsv).

Metrics (standard IR, averaged over the queries)
    P@5      share of the top 5 results that are relevant (grade >= 1)
    R@10     share of the relevant texts found in the top 10
    MRR      1 / rank of the first relevant result
    nDCG@10  graded: a grade-2 text at rank 1 is worth more than a grade-1 text at rank 5

The ablation adds one preprocessing step at a time (text.ANALYZERS), then the
entity layer from Tutorial 2, then query normalisation, so every step's gain is
visible on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .. import config
from .engine import SearchEngine, parse_corpus
from .text import ANALYZERS

QUERIES = config.SEARCH_QUERIES


@dataclass
class Query:
    id: str
    text: str
    kind: str
    relevant: dict[str, int]          # file -> grade


def load_queries(path=QUERIES) -> list[Query]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("id\t"):
            continue
        qid, text, kind, rel = line.split("\t")
        grades = {}
        for item in rel.split():
            f, g = item.split(":")
            grades[f if f.endswith(".txt") else f + ".txt"] = int(g)
        out.append(Query(qid, text, kind, grades))
    return out


def precision_at(ranking: list[str], rel: dict[str, int], k: int) -> float:
    return sum(1 for f in ranking[:k] if f in rel) / k


def recall_at(ranking: list[str], rel: dict[str, int], k: int) -> float:
    return sum(1 for f in ranking[:k] if f in rel) / len(rel) if rel else 0.0


def reciprocal_rank(ranking: list[str], rel: dict[str, int]) -> float:
    return next((1 / r for r, f in enumerate(ranking, 1) if f in rel), 0.0)


def ndcg_at(ranking: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(f, 0) - 1) / math.log2(r + 1) for r, f in enumerate(ranking[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def metrics(ranking: list[str], rel: dict[str, int]) -> dict[str, float]:
    return {"P@5": precision_at(ranking, rel, 5), "R@10": recall_at(ranking, rel, 10),
            "MRR": reciprocal_rank(ranking, rel), "nDCG@10": ndcg_at(ranking, rel, 10)}


def per_query(engine: SearchEngine, queries: list[Query]) -> pd.DataFrame:
    rows = []
    for q in queries:
        ranking = engine.ranking(q.text, 10)
        rows.append({"id": q.id, "query": q.text, "kind": q.kind, **metrics(ranking, q.relevant),
                     "top 3": ", ".join(engine.docs[f].title for f in ranking[:3])})
    return pd.DataFrame(rows)


def systems(nlp, docs=None, kb=None, normaliser=None) -> list[tuple[str, SearchEngine]]:
    """The ablation ladder: each system adds one step to the one before."""
    docs = docs if docs is not None else parse_corpus(nlp)
    out = []
    for a in ANALYZERS:
        out.append((a.name, SearchEngine(nlp, docs, a, entities=False, normalise=False, kb=kb)))
    full = ANALYZERS[-1]
    out.append(("+ entity linking (Tutorial 2)", SearchEngine(nlp, docs, full, entities=True, normalise=False, kb=kb)))
    out.append(("+ query normalisation (full system)",
                SearchEngine(nlp, docs, full, entities=True, normalise=True, kb=kb, normaliser=normaliser)))
    return out


def ablation(nlp, queries: list[Query] | None = None, engines=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overall metrics per system, nDCG@10 per system and query kind)."""
    queries = queries or load_queries()
    engines = engines or systems(nlp)
    overall, by_kind = [], []
    for name, eng in engines:
        pq = per_query(eng, queries)
        overall.append({"system": name, **pq[["P@5", "R@10", "MRR", "nDCG@10"]].mean().round(3).to_dict()})
        by_kind.append({"system": name, **pq.groupby("kind")["nDCG@10"].mean().round(3).to_dict()})
    return pd.DataFrame(overall).set_index("system"), pd.DataFrame(by_kind).set_index("system")
