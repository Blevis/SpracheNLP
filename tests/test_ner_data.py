"""Integrity of the committed NER data, and model-based behaviour of the pipeline."""
import pytest

from sprache import config, corpus
from sprache.ner import bio, experiments, text
from sprache.ner.doc import read_conll


def test_gold_is_valid_bio_and_matches_the_corpus():
    gold = read_conll(config.NER_GOLD)
    assert {s.file for s in gold} == set(experiments.GOLD_FILES)
    texts = corpus.load(config.CORPUS_CLEAN)
    for f in experiments.GOLD_FILES:
        assert [s.words for s in gold if s.file == f] == text.tokenize(texts[f]), f
    for s in gold:
        assert bio.is_valid(s.outer) and bio.is_valid(s.inner)
        assert {t[2:] for t in s.outer + s.inner} <= set(config.ENTITY_TYPES) | {""}
        # every inner entity lies inside an outer one
        for a, b, _ in s.spans("inner"):
            assert any(o[0] <= a and b <= o[1] for o in s.spans())


def test_gold_and_training_texts_do_not_overlap():
    assert not set(experiments.GOLD_FILES) & set(experiments.train_files())
    assert len(experiments.train_files()) == 92


def test_challenge_file_parses():
    items = experiments.read_challenge()
    assert len(items) >= 30
    assert {p for p, _, _ in items} >= {"metonymy", "ambiguity", "nested", "variation", "noisy", "numex", "timex"}


def test_knowledge_base_files_are_consistent():
    import pandas as pd
    kb = pd.read_csv(config.NER_KB).fillna("")
    links = pd.read_csv(config.NER_LINKS).fillna("")
    assert set(kb.type) <= set(config.ENAMEX) | {""}
    assert set(links.file) <= set(corpus.load(config.CORPUS_CLEAN))
    assert (kb.type != "").sum() > 500


@pytest.mark.model
def test_pipeline_phenomena():
    from sprache.analysis import load_model
    from sprache.ner.kb import KnowledgeBase
    from sprache.ner.pipeline import NERPipeline
    from sprache.ner.tagger import EntityTagger, MODEL_FILE
    if not MODEL_FILE.exists():
        pytest.skip("no trained model; run `python -m sprache ner`")
    nlp, kb = load_model(), KnowledgeBase.load()
    pipe = NERPipeline(nlp, kb, EntityTagger(kb).load(), normaliser=experiments._normaliser(kb, experiments.all_files()))

    def found(t, noisy=False):
        return {(m.text, m.type, m.layer) for s in pipe(t, normalise=noisy) for m in s.mentions}

    assert ("Berlin", "ORG", "outer") in found("Berlin lehnt den Vorschlag ab.")                # metonymy
    assert ("Berlin", "GPE", "outer") in found("Wir fahren morgen nach Berlin.")                # literal
    assert ("Wien", "GPE", "inner") in found("Sie studierte an der Universität Wien.")          # nesting
    assert ("21. März 1685", "DATE", "outer") in found("Bach wurde am 21. März 1685 geboren.")  # TIMEX
    assert ("muenchen", "GPE", "outer") in found("ich wohne in muenchen", noisy=True)           # noise
    m = [m for s in pipe("Johann Sebastian Bach wurde in Eisenach geboren. Bachs Werke sind berühmt.")
         for m in s.mentions if m.text == "Bachs"][0]
    assert m.type == "PER" and m.canonical == "Johann Sebastian Bach"                            # variation
