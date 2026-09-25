// Checks the packaged demo before it is published (run by .github/workflows/pages.yml):  node demo/check_site.mjs [site]
// Every file the manifest lists must be present at its recorded size, the model parts must unpack to the recorded
// SHA-256, the ONNX Runtime files must be the recorded version, and the pure-JavaScript engine must answer the page's
// first example from these files. Needs only Node (18 or newer).
import { existsSync, readFileSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const site = resolve(process.argv[2] || fileURLToPath(new URL("./site/", import.meta.url)));
const require = createRequire(import.meta.url);
const core = require(join(site, "rill-core.js"));
const { RillEngine } = require(join(site, "rill-engine.js"));
const fail = (msg) => { console.error(`check_site: ${msg}`); process.exit(1); };
const gunzip = (buf, what) => { try { return gunzipSync(buf); } catch (e) { return fail(`${what} do not unpack (${e.message}): a file is damaged or from another export`); } };

function need(file) {
  const path = join(site, file);
  if (!existsSync(path)) fail(`${file} is missing (the packaged model is committed with the site; see scripts/export_demo.py)`);
  return path;
}
// RillData.put(key, JSON) files; chars is the size the manifest recorded
function data(file, key, chars) {
  const path = need(file), size = statSync(path).size;
  if (chars !== undefined && size !== chars) fail(`${file} is ${size} bytes, the manifest says ${chars}`);
  const { key: k, value } = core.readDataScript(readFileSync(path, "latin1"));
  if (k !== key) fail(`${file} registers "${k}", expected "${key}"`);
  return value;
}

for (const f of ["index.html", "rill-core.js", "rill-engine.js", "engine-worker.js", "THIRD_PARTY_NOTICES.txt"]) need(f);
const manifest = data("model/manifest.js", "manifest");

const gz = Buffer.concat(manifest.parts.map((p) => Buffer.from(data(p.file, p.key, p.chars), "base64")));
const onnx = gunzip(gz, "the model parts");
const sha256 = createHash("sha256").update(onnx).digest("hex");
if (onnx.length !== manifest.bytes || sha256 !== manifest.sha256)
  fail(`the model parts unpack to ${onnx.length} bytes with SHA-256 ${sha256}; the manifest says ${manifest.bytes} bytes, ${manifest.sha256}`);

const rt = manifest.runtime;
const wasm = gunzip(Buffer.from(data(rt.file, rt.key, rt.chars), "base64"), rt.file);
if (wasm.length !== rt.bytes || !WebAssembly.validate(wasm)) fail(`${rt.file} does not hold a valid ${rt.bytes}-byte WebAssembly binary`);
const version = rt.package.slice(rt.package.lastIndexOf("@") + 1);
if (!readFileSync(need("ort/ort.wasm.bundle.min.js"), "utf8").slice(0, 200).includes(`ONNX Runtime Web v${version}`))
  fail(`ort/ort.wasm.bundle.min.js is not the bundle of ${rt.package}`);

// the page's first example, through the pure-JavaScript engine
const tokenizer = new core.BPETokenizer(data(manifest.tokenizer.file, manifest.tokenizer.key, manifest.tokenizer.chars));
const engine = new RillEngine(new Uint8Array(onnx.buffer, onnx.byteOffset, onnx.length), data(manifest.engine.file, manifest.engine.key));
const request = {
  state: "I ordered my new card over two weeks ago and it still hasn't shown up. How much longer should I wait?",
  questions: {
    q1: { type: "choice", instructions: "Which banking intent best describes this customer message?",
          criteria: { card_arrival: "waiting for a new card to arrive", declined_card_payment: "a card payment was declined",
                      lost_or_stolen_card: "card lost or stolen", refund_not_showing_up: "refund has not arrived",
                      change_pin: "change the PIN", exchange_rate: "question about exchange rates" } },
    q2: { type: "noul", instructions: "Is the customer asking for money back?", criteria: null },
    q3: { type: "score", instructions: "How upset is the customer?", criteria: ["not upset", "slightly upset", "upset", "very upset"] }
  }
};
const rec = core.toRecord(request), pk = core.pack(tokenizer, manifest.meta, rec, 1500);
const t0 = performance.now();
const out = engine.run(pk.ids, pk.seg, pk.pos, pk.slots);
const ms = performance.now() - t0;
const cal = core.calibrate(out.slot_logits, out.slot_hidden, pk.layout, manifest.meta.calibrator);
const ans = core.answers(cal.logits, pk.layout, cal.taus);
for (const q of pk.layout) {
  const p = ans[q.qid].p;
  if (!p.every(Number.isFinite) || Math.abs(p.reduce((a, b) => a + b, 0) - 1) > 1e-6) fail(`${q.qid}: probabilities are not a distribution (${p})`);
}
if (ans.q1.choice !== "card_arrival") fail(`the first example was answered ${ans.q1.choice}, expected card_arrival`);
console.log(JSON.stringify({ site, model_mb: +(onnx.length / 1e6).toFixed(1), sha256, runtime: rt.package, tokens: pk.ids.length,
  q1: ans.q1.choice, p_q1: +ans.q1.pmax.toFixed(3), p_yes_q2: +ans.q2.noul.toFixed(3), score_q3: +ans.q3.score.toFixed(2), js_ms: Math.round(ms) }));
