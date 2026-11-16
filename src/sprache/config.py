"""Central configuration: paths, corpus limits, model name, pinned sentences.

Everything that a teammate might want to change lives here, so the rest of
the package (and the notebook) never hard-codes a path or a number.
"""
from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TITLES_FILE = DATA / "titles.txt"
CORPUS_RAW = ROOT / "corpus"            # texts exactly as returned by the API
CORPUS_CLEAN = ROOT / "corpus_clean"    # same texts after clean.clean_text()
SOURCES_CSV = ROOT / "sources.csv"
SKIPPED_CSV = DATA / "skipped.csv"      # titles rejected during the last build
FIGURES = ROOT / "figures"

# --- corpus limits ---------------------------------------------------------
MAX_TEXTS = 100
MIN_WORDS = 100
MAX_WORDS = 300

# --- source ----------------------------------------------------------------
LANG = "de"
API_URL = f"https://{LANG}.wikipedia.org/w/api.php"
LICENCE = "CC BY-SA 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
# Wikimedia asks for a descriptive User-Agent. No personal data on purpose.
USER_AGENT = "SpracheNLP-course-corpus/2.0 (university NLP lab; low volume) python-requests"
BATCH_SIZE = 20          # MediaWiki returns at most 20 intro extracts per request
REQUEST_PAUSE = 1.0      # seconds between batches
MAX_RETRIES = 5

# --- NLP -------------------------------------------------------------------
SPACY_MODEL = "de_core_news_md"
DEMO_SENTENCE = "Johann Wolfgang von Goethe wurde 1749 in Frankfurt am Main geboren."

# Three sentences analysed by hand in section 5 of the notebook. They are
# pinned (not auto-selected) so the written verdicts always match the output.
# Each one is taken verbatim from the corpus; tests check that they still are.
PINNED_SENTENCES = [
    # long proper names, an IPA artefact, coordination of NPs
    "Düsseldorf ([ˈdʏsl̩ˌdɔʁf] , Düsseldorfer Platt Düsseldörp) ist die "
    "Landeshauptstadt Nordrhein-Westfalens und der Behördensitz des "
    "Regierungsbezirks Düsseldorf.",
    # long compound, past-tense verb that is homographic with an adjective
    "Seine Werke beeinflussten spätere Komponistengenerationen und inspirieren "
    "bis heute musikschaffende Künstler zu zahllosen Bearbeitungen.",
    # contraction "im", fronted PP, postposed subject, coordination
    "Neben den Flüssen Spree, Havel und Dahme befinden sich im Stadtgebiet "
    "kleinere Fließgewässer sowie zahlreiche Seen und Wälder.",
]

# --- named-entity recognition (Tutorial 2) ----------------------------------
NER_DIR = DATA / "ner"
NER_LINKS = NER_DIR / "links.csv"          # anchor text -> article -> Wikidata item, per text
NER_KB = NER_DIR / "kb.csv"                # Wikidata item -> type, German label, aliases
NER_SILVER = NER_DIR / "silver.conll"      # automatic BIO annotation of the whole corpus
NER_GOLD = NER_DIR / "gold.conll"          # hand-checked BIO labels for the 8 evaluation texts
NER_CHALLENGE = NER_DIR / "challenge.txt"  # hand-written sentences: metonymy, ambiguity, nesting, noise
NER_OUTPUT = NER_DIR / "entities.csv"      # entity index of the corpus (input for the search engine)
NER_MODELS = ROOT / "models"               # trained CRF model (generated, git-ignored)
SPARQL_URL = "https://query.wikidata.org/sparql"

ENAMEX = ("PER", "ORG", "LOC", "GPE")
NUMEX = ("MONEY", "PERCENT", "QUANTITY")
TIMEX = ("DATE", "TIME")
ENTITY_TYPES = ENAMEX + NUMEX + TIMEX

# Wikidata classes that decide an item's ENAMEX type (instance of / subclass of*).
# The type reached on the shortest path wins; ties go to the earlier entry (Berlin is a city *and* a
# geographic location -> GPE; a university is an organisation and a building -> ORG).
WIKIDATA_CLASSES = {
    # "O": not a named entity of our types (events, periods, movements, works); listed
    # first so that an item that is as close to "event" as to "organisation" is left out
    "O": ["Q1656682", "Q1190554", "Q49773", "Q11514315", "Q17537576", "Q186081", "Q968159"],
    "PER": ["Q5", "Q15632617"],                                  # human, fictional human
    "GPE": ["Q6256", "Q3624078", "Q7275", "Q515", "Q486972",     # country, state, city, settlement
            "Q56061", "Q107390", "Q1520223"],                     # admin. territorial entity, federated state, constituent country
    "ORG": ["Q43229", "Q4830453", "Q783794", "Q2085381",         # organisation, business, company, publisher
            "Q7278", "Q3918", "Q215380"],                         # party, university, musical group
    "LOC": ["Q618123", "Q4022", "Q23397", "Q8502", "Q46831",     # geographic feature, river, lake, mountain, range
            "Q82794", "Q41176", "Q811979", "Q2221906", "Q165"],  # region, building, structure, location, sea
}

# Probes for German-specific tokenizer/lemmatizer behaviour (section 6).
GERMAN_PROBES = [
    "Kraftfahrzeug-Haftpflichtversicherung",
    "Donaudampfschifffahrtsgesellschaft",
    "zum Bahnhof",
    "im Haus",
    "vom Rhein",
    "Straße",
    "Müller-Lüdenscheidt",
    "Er ruft mich morgen an.",
    "z. B. am 3. Oktober 1990",
]
