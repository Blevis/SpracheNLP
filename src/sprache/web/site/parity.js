// Runs queries through the static (JavaScript) engine in Node.js and prints {query: [file, ...]} as JSON.
// Used by sprache.web.export.check_parity:  node parity.js <site dir> '<json list of queries>' <k>
const fs = require("fs");
const path = require("path");
const { createEngine } = require("./static-engine.js");

const [site, queriesJson, k] = process.argv.slice(2);
const read = (name) => JSON.parse(fs.readFileSync(path.join(site, "data", name), "utf8"));
const engine = createEngine(read("search.json"), read("docs.json"));
const out = {};
for (const q of JSON.parse(queriesJson)) out[q] = engine.ranking(q, +k || 10);
process.stdout.write(JSON.stringify(out));
