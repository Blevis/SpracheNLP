"""Unit tests for the NER package: no model, no network."""
import pytest

from sprache import config
from sprache.ner import bio, noise, rules, text
from sprache.ner.context import merge_rules, head_extension
from sprache.ner.kb import Entry, KnowledgeBase, fold, stem_key, strip_genitive, wikitext_links


# ---------------------------------------------------------------- BIO
def test_spans_bio_roundtrip():
    spans = [(0, 3, "PER"), (5, 6, "GPE")]
    tags = bio.spans_to_bio(7, spans)
    assert tags == ["B-PER", "I-PER", "I-PER", "O", "O", "B-GPE", "O"]
    assert bio.bio_to_spans(tags) == spans


def test_invalid_bio_is_read_leniently_and_repaired():
    tags = ["I-PER", "I-PER", "O", "B-GPE", "I-ORG"]
    assert bio.bio_to_spans(tags) == [(0, 2, "PER"), (3, 4, "GPE"), (4, 5, "ORG")]
    assert bio.repair(tags) == ["B-PER", "I-PER", "O", "B-GPE", "B-ORG"]
    assert not bio.is_valid(tags) and bio.is_valid(bio.repair(tags))


def test_nested_spans_go_to_two_layers():
    outer, inner = bio.layers([(4, 6, "ORG"), (5, 6, "GPE"), (0, 1, "PER")])
    assert outer == [(0, 1, "PER"), (4, 6, "ORG")]
    assert inner == [(5, 6, "GPE")]


def test_overlapping_spans_rejected():
    with pytest.raises(ValueError):
        bio.spans_to_bio(4, [(0, 2, "PER"), (1, 3, "ORG")])


def test_nltk_tree_roundtrip():
    words = ["Bach", "lebte", "in", "Leipzig"]
    tags = ["B-PER", "O", "O", "B-GPE"]
    tree = bio.to_tree(words, tags)
    assert [t.label() for t in tree.subtrees()][1:] == ["PER", "GPE"]
    assert bio.from_tree(tree) == tags


# ---------------------------------------------------------------- tokenisation
def test_tokenizer_keeps_abbreviations_and_splits_ranges():
    sents = text.tokenize("Er lebte z. B. in Weimar (1775–1832). Danach nicht.")
    assert len(sents) == 2
    assert "1775" in sents[0] and "–" in sents[0] and "1832" in sents[0]


# ---------------------------------------------------------------- NUMEX / TIMEX
@pytest.mark.parametrize("sentence, expected", [
    ("am 21. März 1685 geboren", [("21. März 1685", "DATE")]),
    ("um 3,5 % auf 12,4 Mrd. Euro", [("3,5 %", "PERCENT"), ("12,4 Mrd. Euro", "MONEY")]),
    ("mit 891 Quadratkilometern", [("891 Quadratkilometern", "QUANTITY")]),
    ("Mitte des 18. Jahrhunderts", [("Mitte des 18. Jahrhunderts", "DATE")]),
    ("im Laufe des 19. und frühen 20. Jahrhunderts", [("19. und frühen 20. Jahrhunderts", "DATE")]),
    ("in den 1960er Jahren", [("1960er Jahren", "DATE")]),
    ("um 14:30 Uhr", [("14:30 Uhr", "TIME")]),
    ("von 1914 bis 1918", [("1914 bis 1918", "DATE")]),
    ("am 2. Februar 962 in Rom", [("2. Februar 962", "DATE")]),
    ("etwa 90 bis 105 Millionen Menschen", [("90 bis 105 Millionen", "QUANTITY")]),
    ("2000 Euro", [("2000 Euro", "MONEY")]),
])
def test_numex_timex(sentence, expected):
    w = sentence.split()
    assert [(" ".join(w[a:b]), t) for a, b, t in rules.spans(w)] == expected


@pytest.mark.parametrize("sentence", ["dort leben 1500 Einwohner", "Herzog Carl August von Weimar"])
def test_numbers_and_names_that_are_not_dates(sentence):
    assert rules.spans(sentence.split()) == []


def test_rule_span_containing_a_name_wins():
    # "Mitte" was tagged GPE (a Berlin district) inside "Mitte September"
    assert merge_rules([(0, 1, "GPE")], [(0, 2, "DATE")]) == [(0, 2, "DATE")]
    # but a name is never cut by a rule span that only overlaps it
    assert merge_rules([(0, 2, "PER")], [(1, 3, "DATE")]) == [(0, 2, "PER")]


# ---------------------------------------------------------------- knowledge base helpers
def test_fold_variants_and_umlauts():
    assert fold("München") == fold("MUENCHEN") == fold("Muenchen") == "muenchen"
    assert fold("Händel") != fold("Handel")
    assert fold("Berlin-Brandenburg") == "berlin brandenburg"


def test_genitive_and_inflection():
    assert strip_genitive("Deutschlands") == "Deutschland"
    assert strip_genitive("Paris") == "Paris" and strip_genitive("Mainz") == "Mainz"
    assert stem_key("Europäischen Union") == stem_key("Europäische Union")


def test_wikitext_links():
    wt = "{{Infobox|x=[[Nope]]}}'''Bach''' wurde in [[Eisenach]] geboren; [[Johann Sebastian Bach|Bach]]s Werke, [[Datei:x.jpg]]"
    assert wikitext_links(wt) == [("Eisenach", "Eisenach"), ("Bachs", "Johann Sebastian Bach")]


@pytest.fixture
def tiny_kb():
    entries = {
        "Q1": Entry("Q1", "GPE", "München", {"München"}),
        "Q2": Entry("Q2", "PER", "Johann Sebastian Bach", {"Johann Sebastian Bach", "Bach"}),
        "Q3": Entry("Q3", "ORG", "Universität Wien", {"Universität Wien"}),
    }
    return KnowledgeBase(entries, __import__("collections").Counter())


def test_kb_lookup(tiny_kb):
    assert tiny_kb.types("Muenchen") == {"GPE"}
    assert tiny_kb.candidates("Bachs")[0].qid == "Q2"
    assert tiny_kb.longest_match(["Johann", "Sebastian", "Bach", "lebte"], 0) == 3
    assert tiny_kb.longest_match(["lebte"], 0) == 0


def test_head_noun_extension():
    assert head_extension(["Landkreis", "München"], 1, "GPE") == ("GPE", True)    # district contains city
    assert head_extension(["Kanton", "Graubünden"], 1, "GPE") == ("GPE", False)  # same entity
    assert head_extension(["Universität", "Wien"], 1, "GPE") == ("ORG", True)
    assert head_extension(["die", "Wien"], 1, "GPE") is None


# ---------------------------------------------------------------- noisy text
def test_corrupt_keeps_token_count():
    words = "Die Stadt München liegt an der Isar .".split()
    for kind in noise.KINDS:
        assert len(noise.corrupt(words, kind, seed=1, rate=1.0)) == len(words)
    assert noise.corrupt(words, "social")[2] == "muenchen"


def test_normaliser_restores_case_umlauts_and_typos():
    norm = noise.Normaliser([["Der", "Rhein", "fließt", "durch", "München", "und", "Basel", "."]] * 3)
    assert norm(["rhein", "fliesst", "durch", "muenchen"]) == ["Rhein", "fließt", "durch", "München"]
    assert norm(["Rehin"]) == ["Rhein"]                     # one transposition away


def test_entity_type_constants():
    assert set(config.ENAMEX) == {"PER", "ORG", "LOC", "GPE"}
    assert set(config.ENTITY_TYPES) == set(config.ENAMEX) | set(config.NUMEX) | set(config.TIMEX)
