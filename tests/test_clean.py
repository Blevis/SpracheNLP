from collections import Counter

import pytest

from sprache.clean import clean_text

# (raw Wikipedia plaintext, expected after cleaning) - taken from the real corpus
CASES = [
    ("Berlin ([bɛr'li:n] ) ist die Hauptstadt.", "Berlin ist die Hauptstadt."),
    ("Frankfurt am Main () ist eine Stadt.", "Frankfurt am Main ist eine Stadt."),
    ("Dresden (; obersorbisch Drježdźany) ist", "Dresden (obersorbisch Drježdźany) ist"),
    ("Düsseldorf ([ˈdʏsl̩ˌdɔʁf] , Düsseldorfer Platt Düsseldörp) ist",
     "Düsseldorf (Düsseldorfer Platt Düsseldörp) ist"),
    ("Die Mosel (französisch , luxemburgisch , lateinisch Mosella) ist",
     "Die Mosel (lateinisch Mosella) ist"),
    ("München (standarddeutsch [ˈmʏnçn̩]  oder [ˈmʏnçən]; bairisch  [ˈmɪŋ(ː)ɐ]) ist",
     "München ist"),
    ("Hamburg ([ˈhambʊʁk] ; regiolektal auch [ˈhambʊɪ̯ç] ; niederdeutsch Hamborg [ˈhambɔʁχ]), amtlich",
     "Hamburg (niederdeutsch Hamborg), amtlich"),
    ("Wien (Aussprache [viːn] ) ist", "Wien ist"),
    ("Ein Computer (englisch; deutsche Aussprache [kɔmˈpjuːtɐ]) oder", "Ein Computer (englisch) oder"),
    ("Berlin unterhält  85 Institute.", "Berlin unterhält 85 Institute."),
]


@pytest.mark.parametrize("raw, expected", CASES)
def test_clean_real_artefacts(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("text", [
    "Die Photosynthese (altgriechisch φῶς phō̂s, deutsch ‚Licht‘) ist ein Prozess.",
    "Er war ein Naturforscher [seiner] Zeit.",
    "Die Garçon-Bar (französisch garçon) liegt in Köln.",
    "Er wurde am 3. Oktober 1990 (Tag der Einheit) geboren; danach zog er um.",
])
def test_clean_leaves_valid_text_alone(text):
    assert clean_text(text) == text


def test_clean_is_idempotent():
    for raw, _ in CASES:
        once = clean_text(raw)
        assert clean_text(once) == once


def test_clean_counts_rule_hits():
    hits = Counter()
    clean_text("Berlin ([bɛr'li:n] ) ist", hits)
    assert hits["ipa_transcription"] == 1 and hits["empty_brackets"] == 1
