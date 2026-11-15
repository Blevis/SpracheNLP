"""Build, store and load the corpus.

Layout produced by :func:`build`::

    corpus/text001.txt ...      raw intros, UTF-8, MIN_WORDS..MAX_WORDS words
    sources.csv                 one row per text: title, URLs, dates, licence, ...
    data/skipped.csv            titles that were rejected, with the reason
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from . import config
from .fetch import Page, fetch_pages

SOURCE_FIELDS = ["file", "topic", "title", "url", "permalink", "revision_id",
                 "retrieved", "last_modified", "licence", "words"]


# --- title list ------------------------------------------------------------
def read_titles(path: Path = config.TITLES_FILE) -> dict[str, list[str]]:
    """Parse titles.txt into {topic: [titles]} (order preserved)."""
    groups: dict[str, list[str]] = {}
    topic = "Allgemein"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("##"):
            topic = line.lstrip("#").strip()
            groups.setdefault(topic, [])
        elif line and not line.startswith("#"):
            groups.setdefault(topic, []).append(line)
    return groups


def round_robin(groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Interleave groups -> [(topic, title), ...] so any prefix is balanced."""
    cols = [[(topic, t) for t in titles] for topic, titles in groups.items()]
    return [item for row in zip_longest(*cols) for item in row if item]


# --- text cutting ----------------------------------------------------------
# Split after . ! ? when the next word starts upper-case, but not after an
# ordinal number ("im 19. Jahrhundert") or a common abbreviation ("z. B.").
_ABBREV = r"(?<!\b[a-z]\.)(?<!\bz\.\sB\.)(?<!\bbzw\.)(?<!\bca\.)(?<!\bSt\.)(?<!\bDr\.)(?<!\bNr\.)(?<!\bu\.\sa\.)(?<!\bv\.\sChr\.)(?<!\bn\.\sChr\.)"
_SENT_END = re.compile(rf"(?<!\b\d\.)(?<!\b\d\d\.){_ABBREV}(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„(])")


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s*\n+\s*", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_END.split(text) if s]


def cut_to_words(text: str, max_words: int = config.MAX_WORDS) -> tuple[str, int]:
    """Keep whole sentences from the start until max_words would be exceeded."""
    out, n = [], 0
    for s in split_sentences(normalise_whitespace(text)):
        w = len(s.split())
        if n + w > max_words:
            break
        out.append(s)
        n += w
    return " ".join(out), n


# --- build -----------------------------------------------------------------
@dataclass
class BuildResult:
    sources: list[dict]
    skipped: list[dict]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(max_texts: int = config.MAX_TEXTS, min_words: int = config.MIN_WORDS,
          max_words: int = config.MAX_WORDS, out_dir: Path = config.CORPUS_RAW,
          titles_file: Path = config.TITLES_FILE, progress: bool = True) -> BuildResult:
    """Fetch, filter and write the raw corpus. Replaces any previous corpus."""
    ordered = round_robin(read_titles(titles_file))
    topic_of = {title: topic for topic, title in ordered}
    pages: list[Page] = fetch_pages([t for _, t in ordered], progress=progress)

    today = dt.date.today().isoformat()
    sources, skipped, seen = [], [], set()
    for page in pages:
        reason = page.problem
        if not reason and page.pageid in seen:
            reason = f"duplicate of '{page.title}' (redirect)"
        text, n = ("", 0) if reason else cut_to_words(page.extract, max_words)
        if not reason and n < min_words:
            reason = f"only {n} words"
        if not reason and len(sources) >= max_texts:
            reason = "MAX_TEXTS reached"
        if reason:
            skipped.append(dict(title=page.requested, reason=reason))
            continue
        seen.add(page.pageid)
        sources.append(dict(
            file=f"text{len(sources) + 1:03d}.txt", topic=topic_of[page.requested],
            title=page.title, url=page.url, permalink=page.permalink,
            revision_id=page.revid, retrieved=today, last_modified=page.timestamp[:10],
            licence=config.LICENCE, words=n, _text=text))

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("text*.txt"):
        old.unlink()
    for row in sources:
        (out_dir / row["file"]).write_text(row.pop("_text"), encoding="utf-8")
    _write_csv(config.SOURCES_CSV if out_dir == config.CORPUS_RAW else out_dir / "sources.csv",
               sources, SOURCE_FIELDS)
    _write_csv(config.SKIPPED_CSV, skipped, ["title", "reason"])
    return BuildResult(sources, skipped)


# --- load ------------------------------------------------------------------
def load(directory: Path = config.CORPUS_RAW) -> dict[str, str]:
    """{filename: text} for every text*.txt in directory, sorted by name."""
    files = sorted(directory.glob("text*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No texts in {directory}. Run `python -m sprache build` first.")
    return {f.name: f.read_text(encoding="utf-8") for f in files}


def read_sources(path: Path = config.SOURCES_CSV) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
