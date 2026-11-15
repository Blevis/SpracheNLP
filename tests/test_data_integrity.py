"""Checks on the committed corpus: run after every `python -m sprache build`."""
import pytest

from sprache import analysis, config, corpus, quality
from sprache.clean import clean_text

pytestmark = pytest.mark.skipif(not config.SOURCES_CSV.exists(), reason="corpus not built")


@pytest.fixture(scope="module")
def raw():
    return corpus.load(config.CORPUS_RAW)


@pytest.fixture(scope="module")
def sources():
    return corpus.read_sources()


def test_sources_match_files(raw, sources):
    assert [r["file"] for r in sources] == list(raw)
    assert len(raw) <= config.MAX_TEXTS


def test_every_text_has_metadata_and_licence(sources):
    for r in sources:
        assert r["title"] and r["url"].startswith("https://de.wikipedia.org/")
        assert r["permalink"].startswith("https://de.wikipedia.org/w/index.php?oldid=")
        assert r["licence"] == config.LICENCE
        assert r["retrieved"] and r["last_modified"]


def test_word_counts_in_range(raw, sources):
    for r in sources:
        n = len(raw[r["file"]].split())
        assert n == int(r["words"])
        assert config.MIN_WORDS <= n <= config.MAX_WORDS


def test_no_duplicate_articles(sources):
    titles = [r["title"] for r in sources]
    assert len(titles) == len(set(titles))


def test_utf8_without_mojibake(raw):
    rep = quality.pattern_report(raw).set_index("check")
    assert rep.loc["mojibake (broken UTF-8)"].files_affected == 0


def test_clean_corpus_is_up_to_date(raw):
    clean = corpus.load(config.CORPUS_CLEAN)
    assert list(clean) == list(raw)
    assert all(clean[n] == clean_text(t) for n, t in raw.items())


@pytest.mark.parametrize("sentence", config.PINNED_SENTENCES)
def test_pinned_sentences_still_in_corpus(raw, sentence):
    assert analysis.locate(sentence, raw), "corpus changed: update PINNED_SENTENCES and section 5"
