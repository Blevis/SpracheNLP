"""Named-entity recognition for German (Tutorial 2).

ENAMEX (PER, ORG, LOC, GPE), NUMEX (MONEY, PERCENT, QUANTITY) and TIMEX
(DATE, TIME) in the BIO scheme, built on NLTK (tokenisation, RegexpParser
chunk grammars, CRFTagger, chunk trees) with spaCy's parser as a helper for
metonymy. See ``pipeline.py`` for how the parts fit together.
"""
