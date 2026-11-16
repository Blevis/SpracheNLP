"""ENAMEX sequence labeller: NLTK's ``CRFTagger`` (linear-chain CRF via python-crfsuite).

A CRF scores the whole BIO sequence, so it learns that I-PER may only follow
B-PER/I-PER and that "Johann Sebastian Bach" is one PER span, which a
token-by-token classifier cannot. Features: see ``features.token_features``.
"""
from __future__ import annotations

import random
from pathlib import Path

import nltk
from nltk.tag import CRFTagger

from .. import config
from . import bio
from .doc import Sentence
from .features import Token, gazetteer_tags, token_features
from .kb import KnowledgeBase

MODEL_FILE = config.NER_MODELS / "enamex.crf.tagger"
TRAINING_OPTIONS = {"c1": 0.05, "c2": 0.05, "max_iterations": 150, "feature.possible_transitions": True}
# Share of training sentences in which the gazetteer feature is hidden. Without it the
# CRF learns "a name is what the gazetteer knows" and misses every unknown name; with
# it, the model also learns from shape, POS and spaCy (feature dropout).
GAZETTEER_DROPOUT = 0.5

# NLTK only reads/writes tool models inside its data roots; register ours.
if str(config.NER_MODELS) not in nltk.data.path:
    nltk.data.path.insert(0, str(config.NER_MODELS))


def enamex_only(tags: list[str]) -> list[str]:
    return [t if t[2:] in config.ENAMEX else "O" for t in tags]


class EntityTagger:
    def __init__(self, kb: KnowledgeBase, use_gazetteer: bool = True, use_spacy: bool = True,
                 options: dict | None = None):
        self.kb = kb
        self.use_gazetteer = use_gazetteer
        self.use_spacy = use_spacy
        self.crf = CRFTagger(feature_func=token_features, training_opt=options or TRAINING_OPTIONS)

    def tokens(self, s: Sentence) -> list[Token]:
        gaz = gazetteer_tags(s.words, self.kb) if self.use_gazetteer else ["O"] * len(s.words)
        tags = s.tags or ["_"] * len(s.words)
        ents = s.ents if self.use_spacy and s.ents else ["O"] * len(s.words)
        return list(zip(s.words, tags, gaz, ents))

    def train(self, sentences: list[Sentence], model_file: Path = MODEL_FILE,
              gazetteer_dropout: float = GAZETTEER_DROPOUT, seed: int = 0) -> "EntityTagger":
        model_file.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        data = []
        for s in sentences:
            if not s.words:
                continue
            toks = self.tokens(s)
            if rng.random() < gazetteer_dropout:
                toks = [(w, t, "O", e) for w, t, _, e in toks]
            data.append(list(zip(toks, enamex_only(s.outer))))
        self.crf.train(data, str(model_file))
        return self

    def load(self, model_file: Path = MODEL_FILE) -> "EntityTagger":
        self.crf.set_model_file(str(model_file))
        return self

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        tagged = self.crf.tag_sents([self.tokens(s) for s in sentences if s.words])
        it = iter(tagged)
        return [bio.repair([t for _, t in next(it)]) if s.words else [] for s in sentences]
