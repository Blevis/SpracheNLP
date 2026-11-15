from sprache.report import esc


def test_esc_special_characters():
    assert esc("a_b & 50% #1 $x") == r"a\_b \& 50\% \#1 \$x"
    assert esc("[ˈdʏsl̩]") == "{[}ˈdʏsl̩{]}"


def test_esc_keeps_spacy_double_dash_lemma():
    # TeX would turn "--" into an en dash; spaCy uses "--" as punctuation lemma
    assert esc("--") == "-{}-"
