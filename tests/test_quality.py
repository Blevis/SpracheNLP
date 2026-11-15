from sprache import quality


def _row(df, check):
    return df.set_index("check").loc[check]


def test_pattern_report_detects_artefacts():
    texts = {"a.txt": "Berlin [bɛr'li:n] () ist  groß. Ã¤ <b>x</b> &amp; [[Link]]",
             "b.txt": "Sauber und ordentlich."}
    rep = quality.pattern_report(texts)
    for check in ["IPA transcription", "empty brackets", "multiple spaces",
                  "mojibake (broken UTF-8)", "HTML tags", "HTML entities", "wiki markup (==, [[, {{)"]:
        assert _row(rep, check).files_affected == 1, check
    assert rep.files_affected.max() == 1


def test_non_latin_and_umlauts():
    rep = quality.pattern_report({"a.txt": "Größe Übermaß", "b.txt": "altgriechisch δῆμος"})
    assert _row(rep, "non-Latin script").files == "b.txt"
    assert _row(rep, "mojibake (broken UTF-8)").files_affected == 0


def test_boilerplate_is_case_sensitive():
    rep = quality.pattern_report({"a.txt": "(siehe auch Gletscherschwund)"})
    assert _row(rep, "boilerplate words").files_affected == 0


def test_repeated_and_english_sentences():
    s = "Das ist ein ganz normaler Satz."
    texts = {"a": f"{s} Anders.", "b": f"{s} This is the house of the king and the queen."}
    assert quality.repeated_sentences(texts) == [(s, 2)]
    assert quality.english_sentences(texts) == [("b", "This is the house of the king and the queen.")]


def test_near_duplicates():
    base = "eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf"
    texts = {"a": base, "b": base + " dreizehn", "c": "ganz etwas anderes steht hier in diesem text"}
    pairs = quality.near_duplicates(texts)
    assert [p[:2] for p in pairs] == [("a", "b")]
