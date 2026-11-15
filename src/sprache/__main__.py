"""Command-line entry point.

    python -m sprache build      fetch corpus/ + sources.csv from Wikipedia (network)
    python -m sprache clean      write corpus_clean/ from corpus/
    python -m sprache check      quality report for raw and clean corpus
    python -m sprache stats      corpus statistics (needs the spaCy model)
    python -m sprache figures    displaCy figures for the pinned sentences
    python -m sprache notebook   execute the notebook in place (nbconvert)
    python -m sprache report     regenerate docs/generated/*.tex (--pdf: compile docs/report.pdf)
    python -m sprache all        build + clean + figures + notebook + report (with PDF)
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


def cmd_notebook(a) -> None:
    nb = config.ROOT / "NLP_Tutorial1_German_Corpus.ipynb"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=900", str(nb)], check=True,
                   cwd=config.ROOT)
    print(f"Executed {nb.name}")


def cmd_report(a) -> None:
    from .report import compile_pdf, generate
    files = generate()
    print(f"Wrote {len(files)} files to docs/generated/")
    if getattr(a, "pdf", False):
        print(f"Compiled {compile_pdf().relative_to(config.ROOT)}")


def cmd_all(a) -> None:
    a.pdf = True
    for step in (cmd_build, cmd_clean, cmd_figures, cmd_notebook, cmd_report):
        print(f"\n### {step.__name__[4:]}")
        step(a)


def main(argv=None) -> None:
    _utf8_stdout()
    p = argparse.ArgumentParser(prog="python -m sprache", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("build", cmd_build), ("clean", cmd_clean), ("check", cmd_check),
                     ("stats", cmd_stats), ("figures", cmd_figures),
                     ("notebook", cmd_notebook), ("report", cmd_report), ("all", cmd_all)]:
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.set_defaults(fn=fn)
        if name == "report":
            sp.add_argument("--pdf", action="store_true", help="also compile docs/report.pdf with LuaLaTeX")
        if name in ("build", "all"):
            sp.add_argument("--max-texts", type=int, default=config.MAX_TEXTS)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
