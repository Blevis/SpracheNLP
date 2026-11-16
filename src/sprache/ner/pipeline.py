"""The full NER pipeline.

    text
     -> NLTK Punkt + word tokenizer (text.py)
     -> [optional] noise normalisation: truecasing, umlauts, typos (noise.py)
     -> spaCy POS / lemma / dependencies on the NLTK tokens (doc.parse)
     -> ENAMEX: NLTK CRFTagger with gazetteer features (tagger.py)
     -> type ambiguity, metonymy, one sense per document (context.py)
     -> NUMEX / TIMEX: NLTK RegexpParser chunk grammar (rules.py)
     -> nested names -> inner BIO layer; canonical names + Wikidata ids
     -> Sentence objects with outer/inner BIO + Mention list (entity index)

Normalisation only ever *replaces* tokens, never splits or merges them, so
the labels found on the normalised view are returned on the original tokens.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import pandas as pd

from .. import config
from . import bio, context, rules, text
from .context import Mention
from .doc import Sentence, parse
from .kb import KnowledgeBase
from .noise import Normaliser
from .tagger import EntityTagger

SPACY_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}      # MISC has no counterpart


class NERPipeline:
    def __init__(self, nlp, kb: KnowledgeBase, tagger: EntityTagger, normaliser: Normaliser | None = None,
                 use_context: bool = True):
        self.nlp, self.kb, self.tagger = nlp, kb, tagger
        self.normaliser = normaliser
        self.use_context = use_context

    # ------------------------------------------------------------------ entry points
    def __call__(self, raw_text: str, file: str = "input", normalise: bool = False) -> list[Sentence]:
        return self.annotate([Sentence(file, w) for w in text.tokenize(raw_text)], normalise)

    def annotate(self, sentences: list[Sentence], normalise: bool = False) -> list[Sentence]:
        """Label sentences in place (outer/inner BIO) and attach ``s.mentions``."""
        view = [copy.copy(s) for s in sentences]
        if normalise and self.normaliser:
            for v in view:
                v.words = self.normaliser(v.words)
        parse(self.nlp, view)
        crf_tags = self.tagger.predict(view)

        # sentence number within its file (for the entity index)
        sent_no, seen = [], {}
        for s in sentences:
            seen[s.file] = seen.get(s.file, -1) + 1
            sent_no.append(seen[s.file])

        mentions: list[Mention] = []
        for k, (s, v, tags) in enumerate(zip(sentences, view, crf_tags)):
            s.tags, s.lemmas, s.deps, s.heads, s.ents = v.tags, v.lemmas, v.deps, v.heads, v.ents
            s.mentions = []
            spans = bio.bio_to_spans(tags)
            if self.use_context:
                fixed = []
                for a, b, t in spans:
                    if not context.plausible(v, a, b, self.kb):
                        continue
                    a = context.trim_descriptor(v, a, b)
                    parts = context.split_title(v, a, b, self.kb) if t == "PER" else None
                    fixed += parts or [(a, b, t)]
                spans = fixed
            for a, b, crf_type in spans:
                m = Mention(s.file, sent_no[k], a, b, s.text(a, b), crf_type, crf_type)
                if self.use_context:
                    votes = context.type_votes(v, a, b, crf_type, self.kb)
                    m.base_type, m.score = votes.most_common(1)[0]
                    m.type, m.reading = context.metonymy(v, a, b, m.base_type)
                if m.type != "O":                       # org-for-product readings are not entities
                    s.mentions.append(m)
            if self.use_context:
                context.coordination_consistency(v, s.mentions, self.kb)
            mentions += s.mentions
        if self.use_context:
            context.one_sense_per_document(mentions)

        for k, (s, v) in enumerate(zip(sentences, view)):
            inner: list[Mention] = []
            if self.use_context:
                # "Landkreis [München]": a head noun before a place name forms the bigger name
                for m in sorted(s.mentions, key=lambda x: x.start):
                    # "[Technische Universität] [München]" -> one ORG with München inside
                    left = next((o for o in s.mentions if o.end == m.start and o is not m), None)
                    if left and m.base_type in context.PLACE_TYPES and left.type in ("ORG", "LOC") \
                            and context.head_class(v.words[left.end - 1]) == left.type:
                        inner.append(Mention(s.file, m.sent, m.start, m.end, m.text, m.type, m.base_type,
                                             layer="inner"))
                        left.end = m.end
                        left.text = s.text(left.start, left.end)
                        s.mentions.remove(m)
                        continue
                    ext = context.head_extension(v.words, m.start, m.base_type)
                    if ext and not left:
                        new_type, keep_inner = ext
                        if keep_inner:
                            inner.append(Mention(s.file, m.sent, m.start, m.end, m.text, m.type, m.base_type,
                                                 m.reading, layer="inner"))
                        m.start -= 1
                        m.text = s.text(m.start, m.end)
                        m.type = m.base_type = new_type
                        m.reading = "literal"
            enamex = {(m.start, m.end, m.type): m for m in s.mentions}
            merged = context.merge_rules(list(enamex), rules.spans(v.words))
            s.mentions = [enamex.get(sp) or Mention(s.file, sent_no[k], sp[0], sp[1], s.text(*sp[:2]), sp[2], sp[2])
                          for sp in merged]
            if self.use_context:
                for m in [m for m in s.mentions if m.type in config.ENAMEX and m.type != "PER" and m.end - m.start > 1]:
                    for a, b, t in context.inner_spans(v, m.start, m.end, self.kb):
                        if not any((i.start, i.end) == (a, b) for i in inner):
                            inner.append(Mention(s.file, m.sent, a, b, s.text(a, b), t, t, layer="inner"))
            outer_spans = [(m.start, m.end, m.type) for m in s.mentions]
            inner = [i for i in inner if any(o[0] <= i.start and i.end <= o[1] and (o[0], o[1]) != (i.start, i.end)
                                             for o in outer_spans)]
            s.outer = bio.spans_to_bio(len(s.words), outer_spans)
            s.inner = bio.spans_to_bio(len(s.words), bio.layers(outer_spans + [(m.start, m.end, m.type) for m in inner])[1])
            s.mentions = sorted(s.mentions + inner, key=lambda m: (m.start, m.layer))
        if self.use_context:
            context.canonicalise([m for s in sentences for m in s.mentions], self.kb)
        return sentences


def mentions_table(sentences: list[Sentence]) -> pd.DataFrame:
    cols = ["file", "sent", "start", "end", "text", "type", "base_type", "reading", "layer", "canonical", "qid"]
    rows = [asdict(m) for s in sentences for m in getattr(s, "mentions", [])]
    return pd.DataFrame(rows, columns=cols + ["score"])[cols]


def spacy_baseline(nlp, sentences: list[Sentence], kb: KnowledgeBase | None = None) -> list[list[str]]:
    """spaCy's own NER on the same tokens, mapped to our tag set (for comparison).

    spaCy's German model knows only PER/ORG/LOC/MISC. With ``kb`` its LOC spans
    are split into GPE/LOC by the knowledge base, which is the fairest comparison.
    """
    from spacy.tokens import Doc
    out = []
    for d in nlp.pipe(Doc(nlp.vocab, words=s.words) for s in sentences):
        spans = []
        for e in d.ents:
            t = SPACY_MAP.get(e.label_)
            if t == "LOC" and kb:
                types = kb.types(e.text)
                t = "GPE" if "GPE" in types or not types else "LOC"
            if t:
                spans.append((e.start, e.end, t))
        out.append(bio.spans_to_bio(len(d), spans))
    return out
