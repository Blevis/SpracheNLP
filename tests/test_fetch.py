"""fetch_batch against a canned API answer (no network)."""
from sprache import fetch

ANSWER = {
    "batchcomplete": True,
    "query": {
        "normalized": [{"from": "köln", "to": "Köln"}],
        "redirects": [{"from": "Bundesrepublik Deutschland", "to": "Deutschland"}],
        "pages": [
            {"pageid": 1, "title": "Köln", "fullurl": "https://de.wikipedia.org/wiki/K%C3%B6ln",
             "revisions": [{"revid": 111, "timestamp": "2026-11-01T10:00:00Z"}], "extract": "Köln ist ..."},
            {"pageid": 2, "title": "Deutschland", "fullurl": "https://de.wikipedia.org/wiki/Deutschland",
             "revisions": [{"revid": 222, "timestamp": "2026-11-02T10:00:00Z"}], "extract": "Deutschland ..."},
            {"pageid": 3, "title": "Harz", "pageprops": {"disambiguation": ""}, "extract": "Harz steht für"},
            {"title": "Gibtsnicht", "missing": True},
        ],
    },
}


def test_fetch_batch_resolves_redirects_and_flags_problems(monkeypatch):
    monkeypatch.setattr(fetch, "_get", lambda session, params: ANSWER)
    pages = fetch.fetch_batch(None, ["köln", "Bundesrepublik Deutschland", "Harz", "Gibtsnicht"])
    koeln, de, harz, missing = pages
    assert (koeln.title, koeln.revid, koeln.problem) == ("Köln", 111, "")
    assert koeln.permalink == "https://de.wikipedia.org/w/index.php?oldid=111"
    assert (de.requested, de.title, de.timestamp[:10]) == ("Bundesrepublik Deutschland", "Deutschland", "2026-11-02")
    assert harz.problem == "disambiguation page"
    assert missing.problem == "missing"


def test_get_retries_on_429(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code, self.headers = code, {"Retry-After": "0"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self):
            return {"query": {}}

    class Session:
        def get(self, *a, **k):
            calls.append(1)
            return Resp(429 if len(calls) < 3 else 200)

    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch._get(Session(), {}) == {"query": {}}
    assert len(calls) == 3
