"""Download article introductions through the official MediaWiki API.

Design notes
* Titles are requested in batches of up to 20 (the API limit for intro
  extracts), which cuts the number of requests by ~20x compared with one
  request per title. That is what caused the HTTP 429 errors in v1.
* 429/5xx answers are retried with exponential back-off, honouring the
  ``Retry-After`` header; ``maxlag`` makes us yield when the servers are busy.
* For every page we also record the latest revision id, so each text can be
  traced to an exact, permanent version of the article (``oldid`` URL).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests

from . import config


@dataclass
class Page:
    requested: str          # title as written in titles.txt
    title: str = ""         # title after redirects/normalisation
    pageid: int | None = None
    revid: int | None = None
    timestamp: str = ""     # ISO time of the latest revision
    url: str = ""
    extract: str = ""
    problem: str = ""       # non-empty -> page is unusable (missing, disambiguation, ...)

    @property
    def permalink(self) -> str:
        if self.revid:
            return f"https://{config.LANG}.wikipedia.org/w/index.php?oldid={self.revid}"
        return self.url


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def _get(session: requests.Session, params: dict) -> dict:
    """GET with retry/back-off for 429, 5xx and API 'maxlag' errors."""
    delay = 2.0
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            r = session.get(config.API_URL, params=params, timeout=30)
        except requests.RequestException:
            if attempt == config.MAX_RETRIES:
                raise
            time.sleep(delay); delay *= 2
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == config.MAX_RETRIES:
                r.raise_for_status()
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait); delay *= 2
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("error", {}).get("code") == "maxlag":
            time.sleep(float(r.headers.get("Retry-After", 5)))
            continue
        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error']}")
        return data
    raise RuntimeError("unreachable")


def _chunks(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fetch_batch(session: requests.Session, titles: list[str]) -> list[Page]:
    """Fetch intro extracts + metadata for up to BATCH_SIZE titles."""
    params = dict(
        action="query", format="json", formatversion=2, maxlag=5,
        prop="extracts|info|revisions|pageprops",
        exintro=1, explaintext=1, exlimit="max",
        inprop="url", rvprop="ids|timestamp", ppprop="disambiguation",
        redirects=1, titles="|".join(titles),
    )
    data = _get(session, params)
    q = data.get("query", {})
    by_title = {p["title"]: p for p in q.get("pages", [])}

    # Large batches can be split by the API ("continue"): merge the missing parts.
    while "continue" in data:
        data = _get(session, {**params, **data["continue"]})
        for p in data.get("query", {}).get("pages", []):
            target = by_title.setdefault(p["title"], p)
            for k, v in p.items():
                target.setdefault(k, v)

    # requested title -> final title (normalisation, then redirects)
    resolve = {t: t for t in titles}
    for key in ("normalized", "redirects"):
        mapping = {m["from"]: m["to"] for m in q.get(key, [])}
        resolve = {req: mapping.get(cur, cur) for req, cur in resolve.items()}

    out = []
    for req in titles:
        p = by_title.get(resolve[req])
        page = Page(requested=req)
        if p is None or p.get("missing") or p.get("invalid"):
            page.problem = "missing"
        elif "disambiguation" in p.get("pageprops", {}):
            page.title, page.problem = p["title"], "disambiguation page"
        else:
            rev = (p.get("revisions") or [{}])[0]
            page.title = p["title"]
            page.pageid = p.get("pageid")
            page.revid = rev.get("revid")
            page.timestamp = rev.get("timestamp", "")
            page.url = p.get("fullurl", "")
            page.extract = p.get("extract", "") or ""
            if not page.extract.strip():
                page.problem = "empty extract"
        out.append(page)
    return out


def fetch_pages(titles: Iterable[str], progress: bool = True) -> list[Page]:
    titles = list(titles)
    session = _session()
    pages: list[Page] = []
    for i, batch in enumerate(_chunks(titles, config.BATCH_SIZE)):
        if i:
            time.sleep(config.REQUEST_PAUSE)
        pages += fetch_batch(session, batch)
        if progress:
            print(f"  fetched {len(pages)}/{len(titles)} titles", flush=True)
    return pages
