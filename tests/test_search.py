"""Search engine: analyzer and compound splitter units, IR metrics, test-query file, and
end-to-end searches on the real corpus (those need the spaCy model)."""
import json
import math

import pytest

from sprache import config, corpus
from sprache.search import evaluate as ev
from sprache.search.text import ANALYZERS, Analyzer, Decompounder, Tok


def tok(text, lemma=None, pos="NOUN", stop=False):
    return Tok(text, lemma or text, pos, stop, 0, len(text), 0)


# ---------------------------------------------------------------- compound splitter
@pytest.fixture(scope="module")
def splitter():
    vocab = {w: w for w in ["recht", "woerterbuch", "markt", "energie", "wende", "bund", "liga", "stadt",
                            "haupt", "heide", "berg", "klima", "wandel"]}
    vocab["weihnacht"] = "weihnachten"          # noun stem -> lemma
    return Decompounder(vocab)


@pytest.mark.parametrize("word, parts", [
    ("energiewende", ("energie", "wende")),
    ("bundesliga", ("bund", "liga")),               # linking element -es-
    ("weihnachtsmarkt", ("weihnacht", "markt")),    # linking element -s-
    ("hauptstadt", ("haupt", "stadt")),
])
def test_split_compounds(splitter, word, parts):
    assert splitter.split(word) == parts


def test_parts_are_lemmas_and_heads(splitter):
    assert splitter.parts("weihnachtsmarkt") == ["weihnachten", "markt"]


def test_head_backoff_for_unknown_modifier(splitter):
    # 'schreib' is unknown, but the head 'woerterbuch' is still found
    assert splitter.parts("rechtschreibwoerterbuch") == ["woerterbuch"]


@pytest.mark.parametrize("word", ["berlin", "heidelberg", "stadt", "klimax"])
def test_no_split_for_short_or_unknown_words(splitter, word):
    # heidel- is not a word, so Heidelberg is not Heidel + Berg
    assert splitter.split(word) == ()


# ---------------------------------------------------------------- analyzer
def test_baseline_keeps_surface_forms_and_stop_words():
    toks = [tok("Die", "der", "DET", True), tok("Flüsse", "Fluss")]
    terms = [t for _, t, _ in ANALYZERS[0].terms(toks)]
    assert terms == ["die", "flüsse"]


def test_full_analyzer_lemmatises_folds_and_drops_stop_words():
    toks = [tok("Die", "der", "DET", True), tok("Flüsse", "Fluss"), tok(",", ",", "PUNCT")]
    assert [t for _, t, _ in Analyzer().terms(toks)] == ["fluss"]


def test_names_keep_their_form_instead_of_a_wrong_lemma():
    # spaCy lemmatises 'Beethoven' as if it were a plural noun
    t = tok("Beethovens", "Beethovens")
    assert Analyzer().term(t, names={"beethoven"}) == "beethoven"
    assert Analyzer().term(tok("Beethoven", "Beethove"), names={"beethoven"}) == "beethoven"


def test_compound_parts_get_lower_weight(splitter):
    out = Analyzer().terms([tok("Energiewende")], splitter)
    assert out[0] == (0, "energiewende", 1.0)
    assert {(t, w) for _, t, w in out[1:]} == {("energie", 0.5), ("wende", 0.5)}


def test_hyphenated_words_give_their_parts():
    out = Analyzer().terms([tok("Fußball-Bund", pos="PROPN")], Decompounder({}))
    assert {t for _, t, _ in out} >= {"fussball", "bund"}


# ---------------------------------------------------------------- metrics
def test_metrics_on_a_small_example():
    rel = {"a": 2, "b": 1}
    ranking = ["x", "a", "y", "b"]
    assert ev.precision_at(ranking, rel, 2) == 0.5
    assert ev.recall_at(ranking, rel, 3) == 0.5
    assert ev.reciprocal_rank(ranking, rel) == 0.5
    assert ev.ndcg_at(["a", "b"], rel, 10) == 1.0
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 + 1 / math.log2(3)
    assert ev.ndcg_at(ranking, rel, 10) == pytest.approx(dcg / idcg)


def test_query_file_points_to_real_texts():
    qs = ev.load_queries()
    files = set(corpus.load(config.CORPUS_CLEAN))
    assert len(qs) >= 25
    assert len({q.id for q in qs}) == len(qs)
    for q in qs:
        assert q.relevant and set(q.relevant) <= files, q.id
        assert set(q.relevant.values()) <= {1, 2} and 2 in q.relevant.values(), q.id


# ---------------------------------------------------------------- end to end (spaCy model)
@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("spacy")
    from sprache.analysis import load_model
    from sprache.search.engine import SearchEngine
    try:
        nlp = load_model()
    except OSError:
        pytest.skip(f"{config.SPACY_MODEL} not installed")
    return SearchEngine(nlp)


@pytest.mark.model
@pytest.mark.parametrize("query, top", [
    ("Flüsse in Deutschland", {"Rhein", "Donau", "Elbe", "Mosel"}),   # lemma: Flüsse -> Fluss
    ("muenchen", {"München"}),                                         # folding
    ("Beethofen", {"Ludwig van Beethoven"}),                           # typo repair
    ("Wörterbuch", {"Duden"}),                                         # inside Rechtschreibwörterbuch
    ("Weihnachten", {"Weihnachtsmarkt"}),
])
def test_search_finds_the_right_text_first(engine, query, top):
    _, results = engine.search(query, 3)
    assert results and results[0]["title"] in top


@pytest.mark.model
def test_query_analysis_links_entities_and_corrects(engine):
    qa = engine.analyse_query("goethe")
    assert qa.corrections == [("goethe", "Goethe")]
    assert any(e["label"] == "Johann Wolfgang von Goethe" for e in qa.entities)


@pytest.mark.model
def test_snippet_marks_the_matching_words(engine):
    _, results = engine.search("Hauptstadt", 1)
    hits = [s["text"] for s in results[0]["snippet"] if s["hit"]]
    assert "Hauptstadt" in hits


@pytest.mark.model
def test_full_system_beats_the_baseline(engine):
    queries = ev.load_queries()
    baseline = ev.per_query(ev.SearchEngine(engine.nlp, engine.docs, ANALYZERS[0], entities=False,
                                            normalise=False, kb=engine.kb), queries)
    full = ev.per_query(engine, queries)
    assert full["nDCG@10"].mean() > baseline["nDCG@10"].mean() + 0.1


@pytest.mark.model
def test_web_api_serialises(engine):
    from sprache.web.server import entity_segments
    d = engine.docs["text005.txt"]
    segs = entity_segments(d.text, d.entities)
    assert "".join(s["text"] for s in segs) == d.text
    assert any(s.get("type") == "GPE" and s["text"] == "München" for s in segs)
    json.dumps(engine.search("Bier", 3)[1], ensure_ascii=False)


# ---------------------------------------------------------------- static site (GitHub Pages)
@pytest.mark.model
def test_static_export_matches_python(engine, tmp_path):
    """The JavaScript engine of the static site ranks the test queries like the Python engine."""
    import shutil
    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    from sprache.web import export
    from sprache.web.server import App
    app = App.__new__(App)                       # reuse the engine fixture instead of loading again
    app.nlp, app.engine, app.load_seconds = engine.nlp, engine, 0
    app._lock, app._systems, app._eval = __import__("threading").Lock(), None, None
    sizes = export.export(tmp_path, app)
    assert (tmp_path / "index.html").read_text(encoding="utf-8").count("static-engine.js") == 1
    assert (tmp_path / ".nojekyll").exists() and sizes["data/search.json"] < 5_000_000
    rows = export.check_parity(tmp_path, app)
    test_queries = {q.text for q in ev.load_queries()}
    assert all(r["python"] == r["js"] for r in rows if r["query"] in test_queries)
    assert all(r["same_top1"] for r in rows)
