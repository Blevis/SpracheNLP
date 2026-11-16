"""Command-line entry point.

    python -m sprache build      fetch corpus/ + sources.csv from Wikipedia (network)
    python -m sprache clean      write corpus_clean/ from corpus/
    python -m sprache check      quality report for raw and clean corpus
    python -m sprache stats      corpus statistics (needs the spaCy model)
    python -m sprache figures    displaCy figures for the pinned sentences
    python -m sprache notebook   execute both notebooks in place (nbconvert)
    python -m sprache report     regenerate docs/generated/*.tex (--pdf: compile docs/report.pdf)

  Named-entity recognition (Tutorial 2)
    python -m sprache ner-kb     download links + Wikidata types into data/ner/ (network)
    python -m sprache ner        silver labels, train the CRF, write data/ner/entities.csv
    python -m sprache ner-eval   evaluation on the gold texts, challenge set, noise, learning curve
    python -m sprache ner-tag    tag your own text: python -m sprache ner-tag "Berlin lehnt ab."

    python -m sprache all        build + clean + figures + ner + notebooks + report (with PDF)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from . import config


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except AttributeError:
            pass


def cmd_build(a) -> None:
    from .corpus import build
    r = build(max_texts=a.max_texts)
    print(f"Saved {len(r.sources)} texts to {config.CORPUS_RAW.name}/ and {config.SOURCES_CSV.name}; "
          f"{len(r.skipped)} titles skipped (see {config.SKIPPED_CSV.relative_to(config.ROOT)}).")


def cmd_clean(a) -> None:
    from collections import Counter
    from .clean import clean_corpus
    rep = clean_corpus()
    total = sum(rep.values(), Counter())
    changed = sum(1 for c in rep.values() if c)
    print(f"Cleaned {len(rep)} texts into {config.CORPUS_CLEAN.name}/ ({changed} changed).")
    for rule, n in total.most_common():
        print(f"  {rule:30s} {n}")


def cmd_check(a) -> None:
    from .corpus import load
    from .quality import pattern_report
    for label, path in (("raw", config.CORPUS_RAW), ("clean", config.CORPUS_CLEAN)):
        if path.exists():
            print(f"\n== {label} ==")
            print(pattern_report(load(path))[["check", "files_affected", "occurrences"]].to_string(index=False))


def cmd_stats(a) -> None:
    from .analysis import load_model
    from .corpus import load
    from .stats import compute, parse
    nlp = load_model()
    st = compute(parse(nlp, load(config.CORPUS_RAW)), nlp.Defaults.stop_words)
    print(st.overview().to_string())
    print("Top 10:", st.top(10))


def cmd_figures(a) -> None:
    from .analysis import load_model, save_displacy
    nlp = load_model()
    for i, s in enumerate(config.PINNED_SENTENCES, 1):
        paths = save_displacy(nlp(s), f"sentence{i}")
        print(f"sentence {i}: {', '.join(p.name for p in paths.values())}")
    save_displacy(nlp(config.DEMO_SENTENCE), "demo")


NOTEBOOKS = ("NLP_Tutorial1_German_Corpus.ipynb", "NLP_Tutorial2_German_NER.ipynb")


def cmd_notebook(a) -> None:
    for name in NOTEBOOKS:
        nb = config.ROOT / name
        if not nb.exists():
            continue
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=1800", str(nb)], check=True,
                       cwd=config.ROOT)
        print(f"Executed {nb.name}")


def cmd_ner_kb(a) -> None:
    from .ner import kb
    links, typed = kb.build()
    print(f"{links} links, {typed} typed Wikidata items -> {config.NER_LINKS.name}, {config.NER_KB.name}")


def cmd_ner(a) -> None:
    from .analysis import load_model
    from .ner import experiments
    df = experiments.build_index(load_model())
    print(f"{len(df)} mentions in {df.file.nunique()} texts -> {config.NER_OUTPUT.relative_to(config.ROOT)}")
    print(df.type.value_counts().to_string())


def cmd_ner_eval(a) -> None:
    import pandas as pd
    from .analysis import load_model
    from .ner import experiments as ex
    pd.set_option("display.width", 200)
    nlp = load_model()
    systems, info = ex.compare_systems(nlp)
    full = info["tables"]["Full pipeline (CRF + context layer + rules)"]
    ch = ex.challenge(nlp, info["tagger"], info["kb"], ex._normaliser(info["kb"], ex.train_files()))
    for title, table in [
        ("systems on the gold texts (F1)", systems.to_string()),
        ("full pipeline per type", full.to_string()),
        ("challenge set", ch.groupby("phenomenon")[["correct", "expected_n", "spurious"]].sum().to_string()),
        ("noisy text (all-type F1)", ex.noise_robustness(nlp, info["tagger"], info["kb"]).to_string()),
        ("learning curve (ENAMEX F1)", ex.learning_curve(nlp).to_string(index=False)),
    ]:
        print(f"\n== {title} ==\n{table}")


def cmd_ner_tag(a) -> None:
    from .analysis import load_model
    from .ner import experiments as ex
    from .ner.kb import KnowledgeBase
    from .ner.pipeline import NERPipeline, mentions_table
    from .ner.tagger import EntityTagger
    if not (config.NER_MODELS / "enamex.crf.tagger").exists():
        raise SystemExit("No trained model yet: run `python -m sprache ner` first.")
    nlp, kb = load_model(), KnowledgeBase.load()
    pipe = NERPipeline(nlp, kb, EntityTagger(kb).load(), normaliser=ex._normaliser(kb, ex.all_files()))
    sents = pipe(" ".join(a.text), normalise=a.noisy)
    for s in sents:
        print("\n".join(f"{w:<20} {o:<12} {i}" for w, o, i in zip(s.words, s.outer, s.inner)) + "\n")
    print(mentions_table(sents).drop(columns=["file"]).to_string(index=False))


def cmd_report(a) -> None:
    from .report import compile_pdf, generate
    files = generate()
    print(f"Wrote {len(files)} files to docs/generated/")
    if getattr(a, "pdf", False):
        print(f"Compiled {compile_pdf().relative_to(config.ROOT)}")


def cmd_all(a) -> None:
    a.pdf = True
    for step in (cmd_build, cmd_clean, cmd_figures, cmd_ner, cmd_notebook, cmd_report):
        print(f"\n### {step.__name__[4:]}")
        step(a)


def main(argv=None) -> None:
    _utf8_stdout()
    p = argparse.ArgumentParser(prog="python -m sprache", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("build", cmd_build), ("clean", cmd_clean), ("check", cmd_check),
                     ("stats", cmd_stats), ("figures", cmd_figures),
                     ("notebook", cmd_notebook), ("report", cmd_report),
                     ("ner-kb", cmd_ner_kb), ("ner", cmd_ner), ("ner-eval", cmd_ner_eval),
                     ("ner-tag", cmd_ner_tag), ("all", cmd_all)]:
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.set_defaults(fn=fn)
        if name == "ner-tag":
            sp.add_argument("text", nargs="+", help="German text to tag")
            sp.add_argument("--noisy", action="store_true", help="normalise case/umlauts/typos first")
        if name == "report":
            sp.add_argument("--pdf", action="store_true", help="also compile docs/report.pdf with LuaLaTeX")
        if name in ("build", "all"):
            sp.add_argument("--max-texts", type=int, default=config.MAX_TEXTS)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
