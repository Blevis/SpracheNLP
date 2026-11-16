"""SpracheNLP - German Wikipedia corpus for NLP Tutorial 1 (Group 2).

Modules
    config    paths, limits, model name, pinned sentences
    fetch     MediaWiki API client (batched, retrying)
    corpus    title list, cutting to MIN..MAX words, writing/loading the corpus
    clean     rule-based removal of Wikipedia plaintext artefacts
    quality   automatic quality checks (raw vs. clean)
    stats     corpus statistics on spaCy Docs
    analysis  token tables, displaCy export, German-specific probes

Command line: ``python -m sprache --help``
"""
__version__ = "2.1.0"
