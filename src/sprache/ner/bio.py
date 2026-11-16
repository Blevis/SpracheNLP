"""BIO scheme: conversion between entity spans, BIO tag sequences and NLTK chunk trees.

    Johann    B-PER        Universität  B-ORG    I-... inner layer:  O
    Sebastian I-PER        München      I-ORG                        B-GPE
    Bach      I-PER

A span is ``(start, end, type)`` with ``end`` exclusive (token indices).
BIO is flat, so nested entities ("Universität [München]") need a second
layer: ``layers()`` puts the outermost spans in layer 0 and the spans inside
them in layer 1, which is the two-column format of the GermEval 2014 task.
"""
from __future__ import annotations

from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tree import Tree

Span = tuple[int, int, str]


def spans_to_bio(n: int, spans: list[Span]) -> list[str]:
    """Spans -> BIO tags. Spans must not overlap (use ``layers`` first)."""
    tags = ["O"] * n
    for s, e, t in sorted(spans):
        if any(tags[i] != "O" for i in range(s, e)):
            raise ValueError(f"overlapping span {(s, e, t)}")
        tags[s] = f"B-{t}"
        for i in range(s + 1, e):
            tags[i] = f"I-{t}"
    return tags


def bio_to_spans(tags: list[str]) -> list[Span]:
    """BIO tags -> spans. An I- tag that does not continue a span of the same
    type starts a new one (the usual lenient CoNLL reading of invalid BIO)."""
    spans, start, typ = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, t = tag.partition("-")
        if start is not None and (prefix != "I" or t != typ):
            spans.append((start, i, typ))
            start = typ = None
        if prefix == "B" or (prefix == "I" and start is None):
            start, typ = i, t
    return spans


def repair(tags: list[str]) -> list[str]:
    """Make a tag sequence valid BIO (every span starts with B-)."""
    return spans_to_bio(len(tags), bio_to_spans(tags))


def is_valid(tags: list[str]) -> bool:
    return list(tags) == repair(tags)


def layers(spans: list[Span]) -> tuple[list[Span], list[Span]]:
    """Split possibly nested spans into an outer and an inner non-overlapping layer.

    Longer spans win the outer layer; a span inside an outer span goes to the
    inner layer; partial overlaps (crossing brackets) are dropped from the inner
    layer, since neither BIO nor a tree can represent them.
    """
    outer: list[Span] = []
    for s in sorted(set(spans), key=lambda x: (x[0] - x[1], x[0])):       # longest first
        if all(s[1] <= o[0] or s[0] >= o[1] for o in outer):
            outer.append(s)
    rest = [s for s in set(spans) if s not in outer]
    inner: list[Span] = []
    for s in sorted(rest, key=lambda x: (x[0] - x[1], x[0])):
        inside = any(o[0] <= s[0] and s[1] <= o[1] for o in outer)
        if inside and all(s[1] <= i[0] or s[0] >= i[1] for i in inner):
            inner.append(s)
    return sorted(outer), sorted(inner)


def to_tree(words: list[str], tags: list[str], pos: list[str] | None = None) -> Tree:
    """BIO -> NLTK chunk tree (S (PER Johann/NE Sebastian/NE Bach/NE) wurde/VAFIN ...)."""
    pos = pos or ["X"] * len(words)
    return conlltags2tree(list(zip(words, pos, tags)))


def from_tree(tree: Tree) -> list[str]:
    """NLTK chunk tree -> BIO tags."""
    return [t for _, _, t in tree2conlltags(tree)]
