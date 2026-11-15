from sprache import corpus


def test_read_titles_groups_and_comments(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("# comment\n## A\nx\ny\n\n## B\nz\n", encoding="utf-8")
    assert corpus.read_titles(p) == {"A": ["x", "y"], "B": ["z"]}


def test_round_robin_balances_topics():
    rr = corpus.round_robin({"A": ["a1", "a2", "a3"], "B": ["b1"], "C": ["c1", "c2"]})
    assert rr == [("A", "a1"), ("B", "b1"), ("C", "c1"), ("A", "a2"), ("C", "c2"), ("A", "a3")]


def test_split_sentences_keeps_ordinals_and_abbreviations():
    text = ("Die Stadt wurde im 19. Jahrhundert gegründet. Am 3. Oktober 1990 kam es z. B. zur Einheit. "
            "Das war 1990. Danach wuchs sie.")
    assert corpus.split_sentences(text) == [
        "Die Stadt wurde im 19. Jahrhundert gegründet.",
        "Am 3. Oktober 1990 kam es z. B. zur Einheit.",
        "Das war 1990.",
        "Danach wuchs sie.",
    ]


def test_cut_to_words_stops_at_sentence_boundary():
    s = "Eins zwei drei vier fünf."
    text, n = corpus.cut_to_words(" ".join([s] * 10), max_words=12)
    assert n == 10 and text == f"{s} {s}"


def test_cut_to_words_normalises_newlines():
    text, n = corpus.cut_to_words("Erster Satz hier.\n\nZweiter Satz da.", max_words=100)
    assert text == "Erster Satz hier. Zweiter Satz da." and n == 6
