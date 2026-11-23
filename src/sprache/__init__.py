"""SpracheNLP - German Wikipedia corpus, NER and search engine (Group 2).

Modules
    config    paths, limits, model name, pinned sentences
    fetch     MediaWiki API client (batched, retrying)
    corpus    title list, cutting to MIN..MAX words, writing/loading the corpus
    clean     rule-based removal of Wikipedia plaintext artefacts
    quality   automatic quality checks (raw vs. clean)
    stats     corpus statistics on spaCy Docs
    analysis  token tables, displaCy export, German-specific probes
    report    LaTeX numbers/tables for docs/report.tex
    ner/      named-entity recognition (Tutorial 2)
    search/   the search engine: German analyzer, BM25 index, evaluation
    web/      local web app: search + presentation (python -m sprache serve)

Command line: ``python -m sprache --help``
"""
__version__ = "3.0.0"
