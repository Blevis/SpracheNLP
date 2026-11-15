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
