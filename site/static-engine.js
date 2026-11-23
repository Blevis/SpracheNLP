"use strict";
/* Static version of the search engine for GitHub Pages.

   The index and all resources are computed in Python (`python -m sprache export-site`)
   and loaded as JSON. This file repeats the *query side* of src/sprache/search/engine.py
   and text.py, function by function, so that the browser answers queries without a server:

     normalise (ner/noise.py)  ->  lemma + POS from the corpus word-form table (instead of spaCy)
     ->  Analyzer.terms with the back-off chain  ->  entity linking (ner/kb.py)
     ->  BM25  ->  snippet

   It also builds the baseline index (surface forms) for the comparison view.
   Works in the browser (window.STATIC_API) and in Node.js (module.exports, for parity tests). */
(function (root) {
  // ------------------------------------------------------------------ text helpers (ner/kb.py)
  const COMBINING = /\p{M}/gu;
  function fold(text) {
    let t = text.toLowerCase().replace(/ß/g, "ss").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue");
    t = t.normalize("NFKD").replace(COMBINING, "");
    return t.replace(/[\s\-‐–]+/g, " ").trim();
  }
  const ENDINGS = ["es", "en", "er", "em", "e", "s", "n"];
  const len = (s) => Array.from(s).length;
  function stemKey(text) {
    return fold(text).split(" ").filter(Boolean).map((w) => {
      for (const e of ENDINGS) if (len(w) > 4 && w.endsWith(e)) return w.slice(0, -e.length);
      return w;
    }).join(" ");
  }
  function stripGenitive(name) {
    if (len(name) > 4 && name.endsWith("s") && !["ss", "us", "is", "os", "as", "x"].some((e) => name.endsWith(e))) return name.slice(0, -1);
    return name;
  }
  const isAlpha = (w) => /^\p{L}+$/u.test(w);
  const isWord = (w) => /[\p{L}\p{N}]/u.test(w);
  const isUpper1 = (w) => { const c = Array.from(w)[0] || ""; return c !== c.toLowerCase() && c === c.toUpperCase(); };

  // Damerau-Levenshtein (optimal string alignment), as nltk.edit_distance(transpositions=True)
  function editDistance(a, b) {
    a = Array.from(a); b = Array.from(b);
    const d = Array.from({ length: a.length + 1 }, (_, i) => { const r = new Array(b.length + 1).fill(0); r[0] = i; return r; });
    for (let j = 0; j <= b.length; j++) d[0][j] = j;
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
    return d[a.length][b.length];
  }

  // query tokenisation: words (with inner hyphens/apostrophes/dots, as spaCy keeps them) and punctuation
  const TOKEN = /[\p{L}\p{N}]+(?:[-'’.][\p{L}\p{N}]+)*-?|[^\s\p{L}\p{N}]/gu;
  const tokenize = (s) => s.match(TOKEN) || [];

  const CONTENT_POS = new Set(["NOUN", "PROPN", "ADJ", "VERB", "ADV", "NUM", "X"]);
  const SPLIT_POS = new Set(["NOUN", "ADJ", "X"]);
  const LINKING = ["", "s", "es", "n", "en", "e", "er"];
  const MIN_PART = 4, HEAD_MIN = 5;
  const INFLECTION = ["nen", "en", "er", "es", "em", "n", "e", "s"];

  function createEngine(S, docs) {
    const postings = S.postings;
    const known = (t) => Object.prototype.hasOwnProperty.call(postings, t);
    const stop = new Set(S.stop);
    const names = new Set(S.names);
    const vocab = S.vocab;
    const inVocab = (w) => Object.prototype.hasOwnProperty.call(vocab, w);
    const lexicon = S.lexicon;
    const byPrefix = {};
    for (const k of Object.keys(lexicon)) if (len(k) >= 5) (byPrefix[k[0]] ??= []).push(k);
    const kb = S.kb;
    const firstTokens = new Set([...Object.keys(kb.index), ...Object.keys(kb.stem)].map((k) => k.split(" ")[0]));
    const docIndex = Object.fromEntries(S.files.map((f, i) => [f, i]));

    // ---------------------------------------------------------------- normaliser (ner/noise.py)
    function normWord(w) {
      if (!(isAlpha(w) || w.includes("-"))) return w;
      const key = fold(w);
      if (Object.prototype.hasOwnProperty.call(lexicon, key)) {
        const best = lexicon[key];
        return !isUpper1(w) || isUpper1(best) ? best : w;
      }
      if (len(key) >= 5) {
        const near = (byPrefix[key[0]] || []).filter((k) => Math.abs(len(k) - len(key)) <= 1 && editDistance(k, key) === 1);
        if (near.length === 1) return lexicon[near[0]];
      }
      return w;
    }

    // ---------------------------------------------------------------- lemma and POS (instead of spaCy)
    function lookup(w) {
      for (const form of [w, w.toLowerCase(), w[0].toUpperCase() + w.slice(1)]) {
        if (Object.prototype.hasOwnProperty.call(S.lex, form)) return S.lex[form];
      }
      if (!isWord(w)) return [w, "PUNCT"];
      if (/^\p{N}+$/u.test(w)) return [w, "NUM"];
      return [w, isUpper1(w) ? "NOUN" : "X"];
    }
    function queryTokens(s) {
      return tokenize(s).map((text) => { const [lemma, pos] = lookup(text); return { text, lemma, pos, stop: stop.has(text.toLowerCase()) }; });
    }

    // ---------------------------------------------------------------- compound splitter (search/text.py)
    const memo = new Map();
    function splitRec(word) {
      if (memo.has(word)) return memo.get(word);
      let best = null, bestKey = null;
      const n = len(word);
      for (let i = MIN_PART; i <= n - MIN_PART; i++) {
        const left = word.slice(0, i);
        if (!inVocab(left)) continue;
        for (const link of LINKING) {
          const rest = word.slice(i);
          if (!rest.startsWith(link)) continue;
          const right = rest.slice(link.length);
          if (len(right) < MIN_PART) continue;
          const tail = inVocab(right) ? [right] : splitRec(right);
          if (!tail) continue;
          const cand = [left, ...tail];
          const key = [cand.length, -len(cand[cand.length - 1]), n - cand.reduce((a, p) => a + len(p), 0)];
          if (!best || key[0] < bestKey[0] || (key[0] === bestKey[0] && (key[1] < bestKey[1] || (key[1] === bestKey[1] && key[2] < bestKey[2])))) {
            best = cand; bestKey = key;
          }
        }
      }
      memo.set(word, best);
      return best;
    }
    function split(word) {
      if (len(word) < 2 * MIN_PART || !isAlpha(word)) return [];
      return splitRec(word) || [];
    }
    function parts(word) {
      const p = split(word);
      if (!p.length) {
        for (let i = MIN_PART; i <= len(word) - HEAD_MIN; i++) if (inVocab(word.slice(i))) return [vocab[word.slice(i)]];
        return [];
      }
      const heads = [];
      for (let i = 1; i < p.length - 1; i++) heads.push(p.slice(i).join(""));
      return [...new Set([...p.map((x) => (inVocab(x) ? vocab[x] : x)), ...heads])];
    }

    // ---------------------------------------------------------------- analyzer (full system)
    function term(t) {
      if (isUpper1(t.text)) for (const form of [stripGenitive(t.text), t.text]) if (names.has(fold(form))) return fold(form);
      return fold(t.lemma);
    }
    function compoundParts(t) {
      const out = [];
      const inner = t.text.replace(/^-+|-+$/g, "");
      if (inner.includes("-")) for (const p of t.text.split("-")) if (len(p) >= 2) out.push(fold(p));
      if (SPLIT_POS.has(t.pos)) out.push(...parts(fold(t.lemma)));
      return [...new Set(out)].filter(Boolean);
    }

    // ---------------------------------------------------------------- knowledge base (ner/kb.py)
    function candidates(name) {
      const keys = new Set([fold(name), fold(stripGenitive(name))]);
      if (name.endsWith("s") && len(name) > 4) keys.add(fold(name.slice(0, -1)));
      let qids = new Set();
      for (const k of keys) for (const q of kb.index[k] || []) qids.add(q);
      if (!qids.size && name.trim().includes(" ")) qids = new Set(kb.stem[stemKey(name)] || []);
      if (!qids.size && len(name) > 5 && name.endsWith("n") && !name.includes(" ")) qids = new Set(kb.index[fold(name.slice(0, -1))] || []);
      const score = (q) => [...keys].reduce((a, k) => a + ((kb.anchors[k] || {})[q] || 0), 0);
      return [...qids].filter((q) => kb.entries[q]).sort((a, b) => score(b) - score(a))
        .map((q) => ({ qid: q, type: kb.entries[q][0], label: kb.entries[q][1] }));
    }
    function longestMatch(words, start) {
      const w = words[start];
      if (![fold(w), fold(stripGenitive(w)), stemKey(w)].some((k) => firstTokens.has(k))) return 0;
      for (let n = Math.min(kb.max_len, words.length - start); n > 0; n--) {
        if (candidates(words.slice(start, start + n).join(" ")).length) return n;
      }
      return 0;
    }
    function link(toks) {
      const words = toks.map((t) => t.text), out = [];
      let i = 0;
      while (i < words.length) {
        const n = isUpper1(words[i]) ? longestMatch(words, i) : 0;
        if (n) {
          const name = words.slice(i, i + n).join(" ");
          const c = candidates(name);
          if (c.length) out.push({ text: name, qid: c[0].qid, type: c[0].type, label: c[0].label });
          i += n;
        } else i++;
      }
      return out;
    }

    // ---------------------------------------------------------------- query analysis (engine.analyse_query)
    function analyseQuery(q) {
      let words = q.split(/\s+/).filter(Boolean);
      const fixed = words.map(normWord);
      const corrections = words.map((w, i) => [w, fixed[i]]).filter(([a, b]) => a !== b);
      words = fixed;
      const normalised = words.join(" ");
      const toks = queryTokens(normalised);
      const byPos = toks.map(() => []);
      toks.forEach((t, i) => {
        if (!isWord(t.text) || t.stop || !CONTENT_POS.has(t.pos)) return;
        let tm = term(t);
        if (!tm) return;
        if (!known(tm) && known(fold(t.text))) tm = fold(t.text);
        // no spaCy here: a word form the corpus never contains gets no lemma from the table,
        // so strip a German inflection ending if that gives a known term (Philosophen -> philosoph)
        if (!known(tm)) {
          const f = fold(t.text);
          const hit = INFLECTION.map((e) => (f.endsWith(e) && len(f) - e.length >= 4 ? f.slice(0, -e.length) : null)).find((c) => c && known(c));
          if (hit) tm = hit;
        }
        byPos[i].push(tm);
        if (!known(tm)) for (const p of compoundParts(t)) if (p !== tm) byPos[i].push(p);
      });
      const terms = new Map();
      for (const lst of byPos) for (const tm of lst) terms.set(tm, 1);
      const entities = link(toks);
      for (const e of entities) terms.set("@" + e.qid, 1);
      return { original: q, normalised, corrections, entities, terms,
        tokens: toks.map((t, i) => ({ word: t.text, lemma: t.lemma, pos: t.pos, stop: t.stop, terms: byPos[i] })) };
    }

    // ---------------------------------------------------------------- BM25 (engine.Index)
    function makeIndex(post, lengths) {
      const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
      const N = lengths.length;
      const idf = (t) => { const n = post[t] ? Object.keys(post[t]).length : 0; return n ? Math.log(1 + (N - n + 0.5) / (n + 0.5)) : 0; };
      function score(qterms, allowed) {
        const { k1, b } = S.params;
        const scores = new Map();
        for (const [t, qw] of qterms) {
          const w = idf(t);
          for (const [d, tf] of Object.entries(post[t] || {})) {
            const di = +d;
            if (allowed && !allowed.has(di)) continue;
            const norm = k1 * (1 - b + b * lengths[di] / avg);
            scores.set(di, (scores.get(di) || 0) + qw * w * tf * (k1 + 1) / (tf + norm));
          }
        }
        return [...scores.entries()].map(([d, s], order) => ({ d, s, order })).sort((x, y) => y.s - x.s || x.order - y.order);
      }
      return { post, idf, score, has: (t, d) => !!(post[t] && post[t][d] != null) };
    }
    const full = makeIndex(postings, S.length);
    const fullTerms = (d) => (i) => docs[d].terms[i] || [];

    // baseline: lower-cased surface forms of every word token, titles x3, no entities
    const basePost = {}, baseLen = [], baseTermsOf = [];
    docs.forEach((doc, d) => {
      const tf = new Map(), byTok = {};
      doc.toks.forEach(([s, e, , w], i) => {
        if (!w) return;
        const t = doc.text.slice(s, e).toLowerCase();
        tf.set(t, (tf.get(t) || 0) + 1); byTok[i] = [t];
      });
      for (const w of doc.title_words) { const t = w.toLowerCase(); tf.set(t, (tf.get(t) || 0) + S.params.title_weight); }
      let total = 0;
      for (const [t, n] of tf) { (basePost[t] ??= {})[d] = n; total += n; }
      baseLen.push(total); baseTermsOf.push(byTok);
    });
    const base = makeIndex(basePost, baseLen);
    function analyseBaseline(q) {
      const toks = tokenize(q);
      const terms = new Map();
      const tokens = toks.map((text) => {
        const t = isWord(text) ? text.toLowerCase() : null;
        if (t) terms.set(t, 1);
        return { word: text, lemma: text, pos: "-", stop: false, terms: t ? [t] : [] };
      });
      return { original: q, normalised: q, corrections: [], entities: [], terms, tokens };
    }

    // ---------------------------------------------------------------- snippets (engine.snippet)
    function findSpan(doc, text) {
      const words = text.split(/\s+/), out = new Set();
      const tw = doc.toks.map(([s, e]) => doc.text.slice(s, e));
      for (let j = 0; j + words.length <= tw.length; j++) {
        if (words.every((w, k) => tw[j + k] === w)) for (let k = 0; k < words.length; k++) out.add(j + k);
      }
      return out;
    }
    function snippet(d, index, termsOf, qterms, qids, maxWords = 45) {
      const doc = docs[d];
      const hits = new Set();
      doc.toks.forEach((_, i) => { if (termsOf(i).some((t) => qterms.has(t))) hits.add(i); });
      for (const q of qids) for (const txt of doc.ent_texts[q] || []) for (const i of findSpan(doc, txt)) hits.add(i);
      const bySent = new Map();
      doc.toks.forEach(([, , s], i) => { if (!bySent.has(s)) bySent.set(s, []); bySent.get(s).push(i); });
      const weight = (s) => bySent.get(s).reduce((a, i) => {
        const idfs = termsOf(i).filter((t) => qterms.has(t)).map(index.idf);
        return a + (idfs.length ? Math.max(...idfs) : 0) + (hits.has(i) ? 1 : 0);
      }, 0);
      const sents = [...bySent.keys()];
      let best = sents[0];
      if (hits.size) {
        let bw = -1;
        for (const s of sents) { const w = weight(s); if (w > bw) { bw = w; best = s; } }
      }
      const full = bySent.get(best);
      let idx = full;
      if (idx.length > maxWords) {
        const first = Math.max(0, idx.findIndex((i) => hits.has(i)));
        const lo = Math.max(0, first - Math.floor(maxWords / 3));
        idx = idx.slice(lo, lo + maxWords);
      }
      const segs = [];
      const start = doc.toks[idx[0]][0];
      if (best !== sents[0] || start > doc.toks[full[0]][0]) segs.push({ text: "… ", hit: false });
      let prev = start;
      for (const i of idx) {
        const [s, e, , w] = doc.toks[i];
        if (s > prev) segs.push({ text: doc.text.slice(prev, s), hit: false });
        segs.push({ text: doc.text.slice(s, e), hit: hits.has(i) && !!w });
        prev = e;
      }
      if (idx[idx.length - 1] !== full[full.length - 1] || best < Math.max(...sents)) segs.push({ text: " …", hit: false });
      return segs;
    }

    // ---------------------------------------------------------------- search (engine.search + server.search)
    function search(q, topic, system, k = 10) {
      const baseline = system === 0;
      const qa = baseline ? analyseBaseline(q) : analyseQuery(q);
      const index = baseline ? base : full;
      const allowed = topic ? new Set(docs.map((doc, i) => (doc.topic === topic ? i : -1)).filter((i) => i >= 0)) : null;
      const ranked = index.score(qa.terms, allowed).slice(0, k);
      const qids = [...qa.terms.keys()].filter((t) => t.startsWith("@")).map((t) => t.slice(1));
      const results = ranked.map(({ d, s }) => {
        const doc = docs[d];
        const termsOf = baseline ? (i) => baseTermsOf[d][i] || [] : fullTerms(d);
        return { file: doc.file, title: doc.title, topic: doc.topic, url: doc.url, permalink: doc.permalink,
          score: Math.round(s * 100) / 100,
          matched: [...qa.terms.keys()].filter((t) => index.has(t, d)).sort(),
          snippet: snippet(d, index, termsOf, qa.terms, baseline ? [] : qids),
          entities: doc.entities };
      });
      return {
        query: q, system: baseline ? "surface forms" : "full system",
        analysis: { normalised: qa.normalised, corrections: qa.corrections, tokens: qa.tokens, entities: qa.entities,
          terms: [...qa.terms].map(([t, w]) => ({ term: t, weight: w, df: index.post[t] ? Object.keys(index.post[t]).length : 0,
            idf: Math.round(index.idf(t) * 100) / 100 })).sort((a, b) => b.idf - a.idf) },
        results,
      };
    }
    return { search, analyseQuery, ranking: (q, k = 10) => search(q, null, null, k).results.map((r) => r.file),
      docIndex, fold, parts };
  }

  // ------------------------------------------------------------------ browser: fake the server API
  function browserApi() {
    const cache = {};
    const load = (name) => (cache[name] ??= fetch("data/" + name).then((r) => {
      if (!r.ok) throw new Error(`could not load data/${name} (${r.status})`);
      return r.json();
    }));
    let engine = null;
    async function getEngine() {
      if (!engine) engine = Promise.all([load("search.json"), load("docs.json")]).then(([S, D]) => createEngine(S, D));
      return engine;
    }
    async function request(path, opts) {
      const u = new URL(path, "http://x");
      const p = u.searchParams;
      switch (u.pathname) {
        case "/api/overview": return load("overview.json");
        case "/api/eval": return load("eval.json");
        case "/api/doc": {
          const docs = await load("docs.json");
          const d = docs.find((x) => x.file === p.get("file"));
          if (!d) throw new Error("not found");
          return d;
        }
        case "/api/search": {
          const e = await getEngine();
          const sys = /^\d+$/.test(p.get("system") || "") ? +p.get("system") : null;
          return e.search((p.get("q") || "").slice(0, 200), p.get("topic") || null, sys);
        }
        case "/api/analyse": {
          const body = JSON.parse((opts && opts.body) || "{}");
          const ex = (await load("pipeline.json")).find((x) => x.text === body.text && x.noisy === !!body.noisy);
          if (ex) return ex.result;
          throw new Error("This is the static version of the site: spaCy and the NER model cannot run in the browser, so only the example texts are available here. Run `python -m sprache serve` for the live pipeline.");
        }
        default: throw new Error("not found");
      }
    }
    return { request, isStatic: true };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { createEngine, fold, editDistance };
  if (typeof window !== "undefined") window.STATIC_API = browserApi();
})(typeof globalThis !== "undefined" ? globalThis : this);
