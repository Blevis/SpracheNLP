"""Tests that need the spaCy model (skipped when it is not installed)."""
import pytest

spacy = pytest.importorskip("spacy")
from sprache import analysis, config, stats  # noqa: E402

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def nlp():
    try:
        return analysis.load_model()
    except OSError:
        pytest.skip(f"{config.SPACY_MODEL} not installed")


def test_stats_definitions(nlp):
    docs = stats.parse(nlp, {"a": "Der Hund und der Hund schlafen.", "b": "Die Katze 2026!"})
    st = stats.compute(docs, nlp.Defaults.stop_words)
    assert st.texts == 2
    assert st.tokens == 11                 # incl. punctuation and the number
    assert st.word_tokens == 8             # alphabetic only
    assert st.unique_words == 6            # der, hund, und, schlafen, die, katze
    assert st.top(1) == [("der", 2)] or st.top(1) == [("hund", 2)]
    assert 0 < st.stopword_share < 1


def test_token_table_and_figures(nlp, tmp_path):
    doc = nlp(config.DEMO_SENTENCE)
    df = analysis.token_table(doc)
    assert list(df.columns) == ["token", "lemma", "pos", "tag", "dep", "head", "ent"]
    paths = analysis.save_displacy(doc, "x", tmp_path)
    assert paths["dep"].read_text(encoding="utf-8").startswith("<svg")
    assert "Goethe" in paths["ent"].read_text(encoding="utf-8")


def test_contractions_stay_single_tokens(nlp):
    probes = analysis.probe_table(nlp, ["zum Bahnhof", "im Haus"])
    assert probes.n_tokens.tolist() == [2, 2]
