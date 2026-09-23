// Latency of the browser runtimes (ONNX Runtime Web, WASM, 1 thread; and the pure-JavaScript engine) on the same
// requests as scripts/bench_latency.py (20 x one question, 20 x three questions), in Node with the page's own code.
//   node demo/bench_wasm.mjs <ort.wasm.bundle.min.mjs> <ort wasm> <lat_ref.json> <out.json>
import { readFileSync, writeFileSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const core = require("./site/rill-core.js");
const { RillEngine } = require("./site/rill-engine.js");
const [ortPath, wasmPath, refPath, outPath] = process.argv.slice(2);
const ort = await import(ortPath);
ort.env.wasm.wasmBinary = readFileSync(wasmPath); ort.env.wasm.numThreads = 1;
const site = new URL("./site/", import.meta.url).pathname;
const data = (path) => core.readDataScript(readFileSync(site + path, "utf8")).value;   // RillData.put(key, JSON) files
const manifest = data("model/manifest.js");
const bytes = await core.decodeModel(manifest.parts.map((p) => data(p.file)), manifest);
const tok = new core.BPETokenizer(data(manifest.tokenizer.file));
const engine = new RillEngine(bytes, data(manifest.engine.file));
const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
const i64 = (a) => BigInt64Array.from(a.map((v) => BigInt(v)));
const ortRun = (pk) => sess.run({ input_ids: new ort.Tensor("int64", i64(pk.ids), [1, pk.ids.length]), segment_ids: new ort.Tensor("int64", i64(pk.seg), [1, pk.ids.length]),
  position_ids: new ort.Tensor("int64", i64(pk.pos), [1, pk.ids.length]), slot_index: new ort.Tensor("int64", i64(pk.slots), [pk.slots.length]) });
const ref = JSON.parse(readFileSync(refPath, "utf8"));
const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const out = { runtime: "onnxruntime-web 1.30.0 WASM (1 thread) and rill-engine.js, Node " + process.version, model_mb: manifest.bytes / 1e6 };
for (const set of ["single", "three"]) {
  const packs = ref[set].map((c) => core.pack(tok, manifest.meta, core.toRecord(c.request), 1500));
  for (const pk of packs.slice(0, 3)) { await ortRun(pk); }            // warm-up, as bench_latency.py
  const tO = [], tJ = [];
  for (const pk of packs) { let t0 = performance.now(); await ortRun(pk); tO.push(performance.now() - t0); }
  for (const pk of packs) { let t0 = performance.now(); engine.run(pk.ids, pk.seg, pk.pos, pk.slots); tJ.push(performance.now() - t0); }
  out[set] = { wasm_median_ms: med(tO), js_median_ms: med(tJ), n: packs.length, median_tokens: med(packs.map((p) => p.ids.length)) };
}
writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
