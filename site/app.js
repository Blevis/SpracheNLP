"use strict";
/* SpracheNLP web app. All data comes from the local Python server (server.py).
   DOM is built with h() and textContent only, so corpus text is never parsed as HTML. */

// ------------------------------------------------------------------ helpers
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") {
      for (const [p, val] of Object.entries(v)) {
        if (p.startsWith("--")) el.style.setProperty(p, val); else el.style[p] = val;
      }
    }
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (s) => document.querySelector(s);
const fmt = (n) => Number(n).toLocaleString("en-US");
const pct = (x) => (x == null ? "–" : x.toFixed(2));
const typeVar = (t) => `var(--${t}, var(--muted))`;

// On GitHub Pages there is no server: static-engine.js (added by `python -m sprache export-site`)
// answers the same API calls in the browser from precomputed JSON.
const STATIC = !!window.STATIC_API;
async function api(path, opts) {
  if (STATIC) return window.STATIC_API.request(path, opts);
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || r.statusText);
  return data;
}
function errorBox(e) { return h("p", { class: "error" }, "Error: " + e.message); }

const cache = {};
async function once(key, fn) { return (cache[key] ??= fn()); }
const overview = () => once("overview", () => api("/api/overview"));
const evaluation = () => once("eval", () => api("/api/eval"));

const ENT_NAMES = { PER: "person", ORG: "organisation", LOC: "location", GPE: "geo-political",
  DATE: "date", TIME: "time", MONEY: "money", PERCENT: "percent", QUANTITY: "quantity" };

// ------------------------------------------------------------------ routing
const views = ["present", "search", "pipeline", "evaluation", "corpus"];
const loaded = {};
function route() {
  const [name, query] = location.hash.slice(1).split("?");
  const view = views.includes(name) ? name : "present";
  for (const v of views) $("#view-" + v).classList.toggle("active", v === view);
  for (const a of document.querySelectorAll("#tabs a")) a.classList.toggle("active", a.dataset.view === view);
  const params = new URLSearchParams(query || "");
  if (view === "search") {
    initSearch();
    if (params.get("q") != null && params.get("q") !== $("#q").value) { $("#q").value = params.get("q"); runSearch(); }
  }
  if (!loaded[view]) {
    loaded[view] = true;
    ({ present: initDeck, pipeline: initPipeline, evaluation: initEval, corpus: initCorpus, search: () => {} })[view]();
  }
}
window.addEventListener("hashchange", route);

function goSearch(q) { location.hash = "search?q=" + encodeURIComponent(q); }

// ------------------------------------------------------------------ theme
(function theme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  if (saved) root.dataset.theme = saved;
  $("#theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
  });
})();

// ------------------------------------------------------------------ search
const EXAMPLES = [
  ["Flüsse in Deutschland", "lemmas: Flüsse = Fluss"],
  ["muenchen bier", "umlaut restored"],
  ["Beethofen", "typo repaired"],
  ["Wörterbuch", "found inside Rechtschreibwörterbuch"],
  ["Weihnachten", "found inside Weihnachtsmarkt"],
  ["Goethe", "entity linked to Wikidata"],
  ["Hauptstadt Deutschlands", "inflected name"],
  ["fussball", "ß folding"],
];
let searchReady = false;
async function initSearch() {
  if (searchReady) return;
  searchReady = true;
  const ex = $("#examples");
  for (const [q, why] of EXAMPLES) ex.append(h("button", { class: "chip", type: "button", title: why, onclick: () => goSearch(q) }, q, h("span", { class: "k" }, why)));
  $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); goSearch($("#q").value.trim()); if (location.hash.includes(encodeURIComponent($("#q").value.trim()))) runSearch(); });
  $("#topic").addEventListener("change", runSearch);
  $("#compare").addEventListener("change", runSearch);
  try {
    const o = await overview();
    for (const [t, n] of o.topics) $("#topic").append(h("option", { value: t }, `${t} (${n})`));
  } catch (e) { /* the search still works without the topic list */ }
}

let searchSeq = 0;
async function runSearch() {
  const q = $("#q").value.trim();
  const out = $("#search-out");
  if (!q) { out.replaceChildren(); return; }
  const seq = ++searchSeq;
  const topic = $("#topic").value;
  const base = `/api/search?q=${encodeURIComponent(q)}&topic=${encodeURIComponent(topic)}`;
  out.replaceChildren(h("div", { class: "loading" }, "Searching ..."));
  try {
    const compare = $("#compare").checked;
    const [full, plain] = await Promise.all([api(base), compare ? api(base + "&system=0") : null]);
    if (seq !== searchSeq) return;
    const left = h("div", {},
      full.analysis.corrections.length ? h("div", { class: "notice" }, "Showing results for ",
        h("b", {}, full.analysis.normalised), " (you typed ", h("i", {}, q), ")") : null,
      compare ? h("div", { class: "columns2", style: { marginTop: "1rem" } },
        h("div", {}, h("h3", {}, "Full system"), resultList(full)),
        h("div", {}, h("h3", {}, "Baseline: surface word forms only"), resultList(plain)))
        : resultList(full));
    out.replaceChildren(h("div", { class: "grid2", style: { marginTop: "1rem" } }, left, queryPanel(full.analysis)));
  } catch (e) { out.replaceChildren(errorBox(e)); }
}

function resultList(data) {
  if (!data.results.length) return h("p", { class: "muted results" }, "No text in the corpus matches this query.");
  return h("div", { class: "results" }, data.results.map((r, i) => h("article", { class: "result" },
    h("button", { class: "title", onclick: () => openDoc(r.file) }, `${i + 1}. ${r.title}`),
    h("div", { class: "meta" }, h("span", {}, r.topic), h("span", {}, r.file), h("span", { title: "BM25 score" }, "score " + r.score),
      h("span", { title: "Index terms that matched" }, "matched: " + r.matched.join(", "))),
    h("div", { class: "snippet" }, r.snippet.map((s) => (s.hit ? h("mark", {}, s.text) : s.text))),
    r.entities.length ? h("div", { class: "ents" }, r.entities.map((e) => h("button", {
      class: "ent", style: { "--c": typeVar(e.type) }, title: `${ENT_NAMES[e.type] || e.type}, Wikidata ${e.qid}, ${e.count}x in this text. Click to search.`,
      onclick: () => goSearch(e.name) }, e.name))) : null)));
}

function queryPanel(a) {
  return h("aside", { class: "panel qa" },
    h("h3", {}, "How the query was read"),
    h("div", { class: "step" }, "1. Normalisation"),
    a.corrections.length ? h("div", {}, a.corrections.map(([x, y]) => h("div", { class: "mono" }, `${x} → ${y}`)))
      : h("div", { class: "small muted" }, "Nothing to correct."),
    h("div", { class: "step" }, STATIC ? "2. Lemma and part of speech (corpus word table)" : "2. spaCy: lemma and part of speech"),
    h("div", { class: "tablewrap" }, h("table", {}, h("tr", {}, h("th", {}, "word"), h("th", {}, "lemma"), h("th", {}, "POS"), h("th", {}, "terms")),
      a.tokens.map((t) => h("tr", { class: t.terms.length ? "" : "stop" }, h("td", {}, t.word), h("td", {}, t.lemma), h("td", { class: "mono" }, t.pos),
        h("td", {}, t.terms.length ? t.terms.map((x) => h("span", { class: "term" }, x)) : h("span", { class: "small" }, t.stop ? "stop word" : "dropped")))))),
    h("div", { class: "step" }, "3. Entity linking"),
    a.entities.length ? a.entities.map((e) => h("div", {}, h("span", { class: "tag", style: { "--c": typeVar(e.type) } }, e.type), " ", e.text, " → ",
      h("a", { href: "https://www.wikidata.org/wiki/" + e.qid, target: "_blank", rel: "noopener" }, `${e.label} (${e.qid})`)))
      : h("div", { class: "small muted" }, "No name from the knowledge base."),
    h("div", { class: "step" }, "4. Index terms (rarest first)"),
    h("div", {}, a.terms.map((t) => h("span", { class: "term" + (t.df ? "" : " zero") + (t.weight < 1 ? " part" : ""),
      title: `in ${t.df} of the texts, idf ${t.idf}` }, t.term))),
    h("p", { class: "small muted", style: { marginTop: ".75rem" } }, "Terms starting with @ are Wikidata items: they match every spelling of the name in the texts. Struck-out terms do not occur in the corpus."));
}

// ------------------------------------------------------------------ document dialog
async function openDoc(file) {
  const dlg = $("#doc");
  $("#doc-title").textContent = "Loading ...";
  $("#doc-meta").textContent = "";
  $("#doc-body").replaceChildren();
  if (!dlg.open) dlg.showModal();
  try {
    const d = await api("/api/doc?file=" + encodeURIComponent(file));
    $("#doc-title").textContent = d.title;
    $("#doc-meta").replaceChildren(`${d.file} · ${d.topic} · ${d.words} words · retrieved ${d.retrieved} · `,
      h("a", { href: d.permalink, target: "_blank", rel: "noopener" }, "revision " + d.revision_id),
      ` · ${d.licence}`, d.cleaned ? " · cleaned (artefacts removed)" : "");
    const used = [...new Set(d.segments.filter((s) => s.type).map((s) => s.type))];
    $("#doc-body").replaceChildren(
      h("div", { class: "legend" }, used.map((t) => h("span", {}, h("span", { class: "tag", style: { "--c": typeVar(t) } }, t), " ", ENT_NAMES[t] || t))),
      h("div", { class: "annotated" }, d.segments.map((s) => (s.type
        ? h("span", { class: "e", style: { "--c": typeVar(s.type) }, title: `${s.type}${s.canonical && s.canonical !== s.text ? " · " + s.canonical : ""}${s.qid ? " · " + s.qid : ""}${s.reading && s.reading !== "literal" ? " · " + s.reading : ""}` },
          s.text, h("span", { class: "tag", style: { "--c": typeVar(s.type) } }, s.type))
        : s.text))),
      h("p", { class: "small muted" }, "Text: German Wikipedia, ", h("a", { href: d.url, target: "_blank", rel: "noopener" }, d.title),
        ", licensed CC BY-SA 4.0. Entities: our Tutorial 2 pipeline (automatic, may contain errors)."));
  } catch (e) { $("#doc-body").replaceChildren(errorBox(e)); }
}
$("#doc-close").addEventListener("click", () => $("#doc").close());
$("#doc").addEventListener("click", (e) => { if (e.target === $("#doc")) $("#doc").close(); });

// ------------------------------------------------------------------ pipeline
const PIPE_EXAMPLES = [
  ["Corpus artefact", "Düsseldorf ([ˈdʏsl̩ˌdɔʁf] , Düsseldorfer Platt Düsseldörp) ist die Landeshauptstadt Nordrhein-Westfalens und der Behördensitz des Regierungsbezirks Düsseldorf."],
  ["Compounds", "Die Donaudampfschifffahrtsgesellschaft und der Weihnachtsmarkt sind typisch deutsche Komposita."],
  ["Entities", "Angela Merkel war von 2005 bis 2021 Bundeskanzlerin. Berlin lehnte den Vorschlag der Europäischen Union am 3. Oktober ab."],
  ["Noisy text", "gestern war ich in muenchen auf dem oktoberfest und habe bier fuer 15 euro getrunken"],
];
function initPipeline() {
  const box = $("#pexamples");
  for (const [label, text] of PIPE_EXAMPLES) box.append(h("button", { class: "chip", type: "button", onclick: () => {
    $("#ptext").value = text; $("#pnoisy").checked = label === "Noisy text"; runPipeline(); } }, label));
  $("#ptext").value = PIPE_EXAMPLES[0][1];
  if (STATIC) {
    $("#ptext").readOnly = true;
    $("#prun").hidden = true;
    $("#ptext").closest(".panel").append(h("p", { class: "notice" },
      "Static version: spaCy and the NER model run in Python, so here the pipeline shows the four example texts, computed in advance. ",
      "For any text of your own, run ", h("code", {}, "python -m sprache serve"), " locally. The search engine on this site works fully."));
    $("#pnoisy").addEventListener("change", runPipeline);
  }
  $("#prun").addEventListener("click", runPipeline);
  runPipeline();
}

async function runPipeline() {
  const out = $("#pipe-out");
  out.replaceChildren(h("div", { class: "loading" }, "Analysing ..."));
  try {
    const raw = $("#ptext").value;
    const d = await api("/api/analyse", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: raw, noisy: $("#pnoisy").checked }) });
    const step = (title, note, ...body) => h("div", { class: "panel stepbox", style: { marginTop: "1rem" } }, h("h3", {}, title), note ? h("p", { class: "small muted" }, note) : null, body);
    out.replaceChildren(
      step("Cleaning", "Rules from Tutorial 1 (clean.py): IPA transcriptions, empty brackets and labels left behind by stripped Wikipedia templates.",
        d.rules.length ? h("div", {}, h("p", {}, "Rules that fired: ", d.rules.map(([r, n]) => h("span", { class: "term" }, `${r} ×${n}`))),
          h("div", { class: "example diff" }, diffWords(raw, d.clean)))
          : h("p", { class: "muted" }, "Nothing to clean: the text has none of the known artefacts.")),
      step("spaCy de_core_news_md: tokens, lemmas, POS, dependencies", `${d.tokens.length} tokens in ${d.sentences.length} sentence(s). Grey rows are stop words or punctuation; they are not indexed.`,
        h("div", { class: "tablewrap" }, h("table", {},
          h("tr", {}, ["token", "lemma", "POS", "tag", "dep", "head", "index terms"].map((x) => h("th", {}, x))),
          d.tokens.map((t) => h("tr", { class: t.terms.length ? "" : "stop" }, h("td", {}, t.text), h("td", {}, t.lemma), h("td", { class: "mono" }, t.pos),
            h("td", { class: "mono" }, t.tag), h("td", { class: "mono" }, t.dep), h("td", {}, t.head),
            h("td", {}, t.terms.map((x) => h("span", { class: "term" + (x.weight < 1 ? " part" : ""), title: x.weight < 1 ? "compound part, weight " + x.weight : "" }, x.term)))))))),
      d.ner ? step("Named entities (Tutorial 2): ENAMEX, NUMEX, TIMEX in BIO", "CRF tagger with knowledge-base features, context rules and a chunk grammar for amounts and dates. Two BIO layers: outer names and names nested inside them.",
        h("div", { class: "annotated" }, nerText(d.ner)),
        h("div", { class: "tablewrap", style: { marginTop: ".75rem" } }, d.ner.sentences.map((s) => h("table", { style: { marginBottom: ".75rem" } },
          h("tr", {}, h("th", {}, "word"), s.normalised ? h("th", {}, "normalised") : null, h("th", {}, "outer BIO"), h("th", {}, "inner BIO")),
          s.words.map((w, i) => h("tr", {}, h("td", {}, w), s.normalised ? h("td", { class: s.normalised[i] !== w ? "" : "muted" }, s.normalised[i]) : null,
            bioCell(s.outer[i]), bioCell(s.inner[i]))))))) : step("Named entities", null, h("p", { class: "muted" }, "No trained NER model yet. Run python -m sprache ner once, then restart the server.")));
  } catch (e) { out.replaceChildren(errorBox(e)); }
}
function bioCell(tag) {
  const t = tag.split("-")[1];
  return h("td", { class: "mono bio-" + tag[0], style: t ? { color: typeVar(t) } : null }, tag);
}
function nerText(ner) {
  const parts = [];
  for (const s of ner.sentences) {
    let i = 0;
    while (i < s.words.length) {
      const tag = s.outer[i];
      if (tag.startsWith("B-")) {
        const t = tag.slice(2); let j = i + 1;
        while (j < s.words.length && s.outer[j] === "I-" + t) j++;
        parts.push(h("span", { class: "e", style: { "--c": typeVar(t) }, title: ENT_NAMES[t] || t }, s.words.slice(i, j).join(" "), h("span", { class: "tag", style: { "--c": typeVar(t) } }, t)), " ");
        i = j;
      } else { parts.push(s.words[i] + " "); i++; }
    }
  }
  return parts;
}
function diffWords(a, b) {
  // character-level diff (LCS) between the raw and the cleaned text: removed characters are struck out.
  // The cleaning rules only delete, so the cleaned text is a subsequence of the raw text.
  const x = [...a], y = [...b];
  const n = x.length, m = y.length;
  if (n * m > 2_000_000) return b;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const ops = []; let i = 0, j = 0;
  const push = (kind, ch) => { const last = ops[ops.length - 1]; if (last && last.kind === kind) last.text += ch; else ops.push({ kind, text: ch }); };
  while (i < n && j < m) {
    if (x[i] === y[j]) { push("same", x[i]); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { push("del", x[i]); i++; }
    else { push("ins", y[j]); j++; }
  }
  while (i < n) push("del", x[i++]);
  while (j < m) push("ins", y[j++]);
  return ops.filter((o) => !(o.kind === "del" && !o.text.trim())).map((o) => (o.kind === "del" ? h("del", {}, o.text) : o.kind === "ins" ? h("b", {}, o.text) : o.text));
}

// ------------------------------------------------------------------ evaluation
function bars(rows, key, max = 1) {
  return h("div", { class: "bars" }, rows.map((r, i) => {
    const prev = i ? rows[i - 1][key] : null;
    const d = prev == null ? null : r[key] - prev;
    return [h("div", {}, r.system), h("div", { class: "bar" }, h("div", { style: { width: (100 * r[key] / max) + "%" } })),
      h("div", { class: "num" }, pct(r[key]), d != null && Math.abs(d) >= 0.005 ? h("span", { class: "small delta " + (d > 0 ? "up" : "down") }, " " + (d > 0 ? "+" : "") + d.toFixed(2)) : null)];
  }));
}
async function initEval() {
  const out = $("#eval-out");
  try {
    const e = await evaluation();
    const kinds = Object.keys(e.by_kind[0]).filter((k) => k !== "system");
    const rel = Object.fromEntries(e.queries.map((q) => [q.id, q.relevant]));
    out.replaceChildren(h("div", { class: "stack" },
      h("div", { class: "panel" }, h("h2", {}, "What each preprocessing step adds"),
        h("p", { class: "muted" }, `${e.per_query.length} test queries with hand-made relevance judgments (data/search/queries.tsv). Each row adds one step to the row above. Metric: nDCG@10, 1.0 means every relevant text is ranked at the top.`),
        bars(e.overall, "nDCG@10")),
      h("div", { class: "panel" }, h("h3", {}, "All metrics"), h("div", { class: "tablewrap" }, h("table", {},
        h("tr", {}, h("th", {}, "system"), ["P@5", "R@10", "MRR", "nDCG@10"].map((k) => h("th", { class: "num" }, k))),
        e.overall.map((r) => h("tr", {}, h("td", {}, r.system), ["P@5", "R@10", "MRR", "nDCG@10"].map((k) => h("td", { class: "num" }, pct(r[k])))))))),
      h("div", { class: "panel" }, h("h3", {}, "nDCG@10 by kind of query"),
        h("p", { class: "small muted" }, "inflection: the query form differs from the text (Flüsse / Fluss). compound: the word sits inside a compound. spelling: no umlauts, lower case or a typo. entity: a name written differently in the texts."),
        h("div", { class: "tablewrap" }, h("table", { class: "heat" },
          h("tr", {}, h("th", {}, "system"), kinds.map((k) => h("th", { class: "num" }, k))),
          e.by_kind.map((r) => h("tr", {}, h("td", {}, r.system), kinds.map((k) => h("td", { class: "num", style: { background: `color-mix(in srgb, var(--accent) ${Math.round(r[k] * 35)}%, transparent)` } }, pct(r[k])))))))),
      h("div", { class: "panel" }, h("h3", {}, "Every query (full system vs. baseline)"),
        h("div", { class: "tablewrap" }, h("table", {},
          h("tr", {}, ["id", "query", "kind", "baseline", "full", "top 3 results", "relevant texts"].map((x) => h("th", {}, x))),
          e.per_query.map((q) => h("tr", {}, h("td", { class: "mono" }, q.id),
            h("td", {}, h("a", { href: "#search?q=" + encodeURIComponent(q.query) }, q.query)), h("td", {}, q.kind),
            h("td", { class: "num" }, pct(q["baseline nDCG@10"])), h("td", { class: "num" }, h("b", {}, pct(q["nDCG@10"]))),
            h("td", { class: "small" }, q["top 3"]),
            h("td", { class: "small" }, rel[q.id].map((r) => `${r.title} (${r.grade})`).join(", "))))))),
      h("div", { class: "panel" }, h("h3", {}, "How to read this"),
        h("ul", {}, h("li", {}, "The query set is small (30 queries) and was written and judged by us, the people who built the system, so the absolute numbers are optimistic. The differences between the steps are the meaningful part."),
          h("li", {}, "Stop-word removal changes almost nothing: BM25's idf already gives words like der or und a weight close to zero."),
          h("li", {}, "Umlaut and case folding has the largest single effect because three test queries are typed without umlauts; query normalisation then repairs the remaining spelling queries (the typo Beethofen)."),
          h("li", {}, "Known failure: Radioaktivität does not find radioaktiv. That is derivation, not inflection, so neither the lemmatizer nor the compound splitter connects them.")))));
  } catch (err) { out.replaceChildren(errorBox(err)); }
}

// ------------------------------------------------------------------ corpus
async function initCorpus() {
  const out = $("#corpus-out");
  try {
    const o = await overview();
    out.replaceChildren(h("div", { class: "stack" },
      h("div", { class: "panel" }, h("h2", {}, "The corpus"),
        h("p", { class: "muted" }, `${o.texts} introductions of German Wikipedia articles, ${fmt(o.words)} words, ${o.topics.length} topics, ${o.licence}, retrieved ${o.retrieved}. Click a title to read it with its named entities.`),
        h("div", { class: "doclist" }, o.documents.reduce((acc, d) => {
          let sec = acc.find((s) => s.topic === d.topic);
          if (!sec) acc.push(sec = { topic: d.topic, items: [] });
          sec.items.push(d); return acc;
        }, []).map((s) => h("section", {}, h("h3", {}, s.topic), s.items.map((d) => h("button", { onclick: () => openDoc(d.file) }, d.title))))))));
  } catch (e) { out.replaceChildren(errorBox(e)); }
}

// ------------------------------------------------------------------ presentation
let slide = 0, slides = [];
async function initDeck() {
  const deck = $("#deck");
  let o;
  try { o = await overview(); } catch (e) { deck.replaceChildren(errorBox(e)); return; }
  slides = buildSlides(o);
  deck.replaceChildren(...slides);
  $("#dots").replaceChildren(...slides.map((_, i) => h("i", { onclick: () => show(i), title: "Slide " + (i + 1) })));
  show(0);
}
function show(i) {
  slide = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, k) => s.classList.toggle("active", k === slide));
  [...$("#dots").children].forEach((d, k) => d.classList.toggle("on", k === slide));
  $("#counter").textContent = `${slide + 1} / ${slides.length}`;
  const s = slides[slide];
  if (s.dataset.lazy && !s.dataset.done) { s.dataset.done = "1"; s.lazy(); }
}
$("#prev").addEventListener("click", () => show(slide - 1));
$("#next").addEventListener("click", () => show(slide + 1));
$("#fs").addEventListener("click", toggleFs);
function toggleFs() {
  document.body.classList.toggle("fullscreen");
  if (document.body.classList.contains("fullscreen")) document.documentElement.requestFullscreen?.().catch(() => {});
  else if (document.fullscreenElement) document.exitFullscreen?.();
}
document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) document.body.classList.remove("fullscreen"); });
document.addEventListener("keydown", (e) => {
  if (!$("#view-present").classList.contains("active") || e.target.closest("input, textarea, select") || $("#doc").open) return;
  if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); show(slide + 1); }
  if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); show(slide - 1); }
  if (e.key === "f" || e.key === "F") toggleFs();
});

function S(kicker, title, lead, ...body) {
  return h("section", { class: "slide" }, h("div", { class: "kicker" }, kicker), h("h2", {}, title),
    lead ? h("p", { class: "lead" }, lead) : null, h("div", { class: "body" }, body));
}
function stat(v, label) { return h("div", { class: "stat" }, h("b", {}, v), h("span", {}, label)); }
function flow(nodes) {
  const out = [];
  nodes.forEach(([b, s], i) => { if (i) out.push(h("div", { class: "arrow" }, "→")); out.push(h("div", { class: "node" }, h("b", {}, b), h("span", {}, s))); });
  return h("div", { class: "flow" }, out);
}

function buildSlides(o) {
  const maxTopic = Math.max(...o.topics.map(([, n]) => n));
  const byType = Object.fromEntries(o.entities.by_type);
  const s = [];

  s.push(h("section", { class: "slide" },
    h("div", { class: "kicker" }, "NLP project · Group 2 · German"),
    h("h1", {}, "SpracheNLP: a search engine for German Wikipedia texts"),
    h("p", { class: "lead" }, "We collected a German corpus, prepared it with NLP preprocessing, added named-entity recognition, and used all of it to build a search engine that understands German word forms, compounds, umlauts and names."),
    h("div", { class: "body" },
      h("div", { class: "stats" }, stat(o.texts, "texts"), stat(fmt(o.words), "words"), stat(o.topics.length, "topics"), stat(fmt(o.entities.mentions), "entity mentions")),
      h("p", { class: "muted" }, "Blevis Allushi · Darsi Ramaj · Geri Belba"),
      h("p", {}, h("a", { href: "#search" }, "Skip to the live search")))));

  s.push(S("1 · Corpus collection", "100 Wikipedia introductions, 8 balanced topics",
    `Fetched through the official MediaWiki API, cut at a sentence boundary to ${o.min_words}–${o.max_words} words (median ${o.median_words}). Every text is pinned to its exact revision and carries its licence (${o.licence}).`,
    h("div", { class: "bars", style: { gridTemplateColumns: "minmax(150px, 260px) 1fr 3rem" } }, o.topics.map(([t, n]) => [h("div", {}, t),
      h("div", { class: "bar" }, h("div", { style: { width: (100 * n / maxTopic) + "%" } })), h("div", { class: "num" }, n)])),
    h("ul", {}, h("li", {}, "Batched API requests with retry and back-off: our first version lost 14 of 34 titles to rate limiting; this one loses none."),
      h("li", {}, "Round-robin selection across topics, so the corpus stays balanced at any size."),
      h("li", {}, "Source list with title, URL, revision link, dates and licence: sources.csv."))));

  s.push(S("2 · Preprocessing", "Cleaning, then the German spaCy pipeline",
    `Automatic quality checks found a systematic artefact: Wikipedia templates that the API strips but whose brackets and labels stay behind. ${o.cleaning.changed} of ${o.texts} texts needed cleaning.`,
    h("div", { class: "example" }, "Düsseldorf ([ˈdʏsl̩ˌdɔʁf] , Düsseldorfer Platt Düsseldörp) ist die Landeshauptstadt ..."),
    h("div", {}, o.cleaning.rules.map(([r, n]) => h("span", { class: "term" }, `${r} ×${n}`))),
    flow([["Tokenisation", "spaCy, NLTK Punkt"], ["POS + morphology", "STTS tags"], ["Lemmatisation", "Flüsse → Fluss"], ["Dependencies", "TIGER labels"], ["Stop words", `${o.model}`]]),
    h("p", {}, h("a", { href: "#pipeline" }, "Try the pipeline on your own text"))));

  s.push(S("2 · Preprocessing for German", "What makes German hard, and what we do about it", null,
    h("div", { class: "columns2" },
      h("div", {}, h("h3", {}, "Problem"), h("ul", {},
        h("li", {}, "Inflection: Fluss, Flusses, Flüsse, Flüssen"),
        h("li", {}, "Compounds: Rechtschreibwörterbuch is one token"),
        h("li", {}, "Umlauts and ß, often typed as ae, oe, ue, ss"),
        h("li", {}, "Every noun is capitalised, so case does not mark names"),
        h("li", {}, "Names have many forms: Goethe, Johann Wolfgang von Goethe, Goethes"))),
      h("div", {}, h("h3", {}, "Our step"), h("ul", {},
        h("li", {}, "spaCy lemmas (names keep their form: Beethoven, not Beethove)"),
        h("li", {}, `Corpus-driven compound splitter with linking elements (Weihnacht-s-markt); ${fmt(o.index.compounds_split)} compound tokens split`),
        h("li", {}, "Folding: München = Muenchen = muenchen"),
        h("li", {}, "Truecasing, umlaut restoration and typo repair from the corpus lexicon"),
        h("li", {}, "Entity linking to Wikidata items through our knowledge base"))))));

  s.push(S("Tutorial 2 · Named entities", "ENAMEX, NUMEX and TIMEX in BIO, without hand-labelled training data",
    "A CRF tagger trained on automatic (silver) labels from Wikipedia links and Wikidata types, plus context rules for ambiguity and metonymy and a chunk grammar for amounts and dates.",
    h("div", { class: "tablewrap" }, h("table", {}, h("tr", {}, h("th", {}, "F1 on 8 hand-checked texts"), ["names", "amounts", "dates", "all"].map((x) => h("th", { class: "num" }, x))),
      o.ner_f1.map((r) => h("tr", {}, h("td", {}, r.system), [r.ENAMEX, r.NUMEX, r.TIMEX, r.all].map((v) => h("td", { class: "num" }, pct(v))))))),
    h("div", { class: "stats" }, ["PER", "ORG", "GPE", "LOC", "DATE", "QUANTITY"].map((t) => h("div", { class: "stat", style: { borderColor: typeVar(t) } }, h("b", { style: { color: typeVar(t) } }, fmt(byType[t] || 0)), h("span", {}, ENT_NAMES[t])))),
    h("p", { class: "muted small" }, `${fmt(o.entities.linked)} of ${fmt(o.entities.mentions)} mentions are linked to a Wikidata item. Those links become search terms.`)));

  s.push(S("3 · Application", "A search engine for the corpus",
    "BM25 ranking over an inverted index. Documents and queries go through the same German analysis, so they meet in the same term space.",
    h("h3", {}, "Indexing (once)"),
    flow([["Clean text", "corpus_clean/"], ["spaCy", "lemma, POS"], ["Terms", "fold + compounds"], ["Entities", "@Wikidata ids"], ["Inverted index", `${fmt(o.index.terms)} terms + ${fmt(o.index.entity_terms)} entities`]]),
    h("h3", {}, "Every query"),
    flow([["Normalise", "muenchen → München"], ["spaCy", "lemma, POS"], ["Same analyzer", "back-off for unknown words"], ["Link names", "Goethe → Q5879"], ["BM25 + snippet", "matching words marked"]])));

  const demo = S("3 · Live demo", "Try it", "Each example shows one preprocessing step at work. The right-hand panel explains how the query was read.",
    h("div", { class: "chips" }, EXAMPLES.map(([q, why]) => h("button", { class: "chip", onclick: () => goSearch(q) }, q, h("span", { class: "k" }, why)))),
    h("p", {}, "Tick ", h("i", {}, "Compare with the baseline"), " on the search page to see plain word matching next to the full system."));
  s.push(demo);

  const ev = S("Evaluation", "What each step adds", "30 test queries with hand-made relevance judgments; nDCG@10 as each step is added.", h("div", { class: "loading" }, "Running the evaluation ..."));
  ev.dataset.lazy = "1";
  ev.lazy = async () => {
    const body = ev.querySelector(".body");
    try {
      const e = await evaluation();
      const first = e.overall[0]["nDCG@10"], last = e.overall[e.overall.length - 1]["nDCG@10"];
      body.replaceChildren(bars(e.overall, "nDCG@10"),
        h("p", {}, `From ${pct(first)} with plain word forms to ${pct(last)} with the full system. `, h("a", { href: "#evaluation" }, "Details per query")));
    } catch (err) { body.replaceChildren(errorBox(err)); }
  };
  s.push(ev);

  s.push(S("Discussion", "What we learned", null,
    h("ul", {},
      h("li", {}, "Data quality first: the cleaning rules came from looking at real artefacts, and every rule is tested on real examples."),
      h("li", {}, "For German search, lemmas, folding and compound splitting matter far more than stop-word removal."),
      h("li", {}, "The NER work pays off twice: entity chips on every result, and name variants matched through Wikidata."),
      h("li", {}, "Limits: 100 encyclopedic texts are not everyday German; the test queries were judged by us; derivation (Radioaktivität / radioaktiv) is not handled."),
      h("li", {}, "Everything is reproducible: one command rebuilds the corpus, the notebooks and the report; over 100 tests guard it.")),
    h("p", { class: "muted" }, "Code, notebooks and report: see the README of the repository.")));
  return s;
}

// ------------------------------------------------------------------ start
route();
