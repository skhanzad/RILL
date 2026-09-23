// Check the pure-JavaScript engine (site/rill-engine.js) against ONNX Runtime Web on the same 8-bit model, and both
// against PyTorch reference probabilities:  node demo/engine_check.mjs <ort.wasm.bundle.min.mjs> <ort wasm> <ref.json> [n]
import { readFileSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const core = require("./site/rill-core.js");
const { RillEngine } = require("./site/rill-engine.js");
const [ortPath, wasmPath, refPath, nMax] = process.argv.slice(2);
const ort = await import(ortPath);
ort.env.wasm.wasmBinary = readFileSync(wasmPath); ort.env.wasm.numThreads = 1;
const site = new URL("./site/", import.meta.url).pathname;
const data = (path) => core.readDataScript(readFileSync(site + path, "utf8")).value;   // RillData.put(key, JSON) files
const manifest = data("model/manifest.js");
const bytes = await core.decodeModel(manifest.parts.map((p) => data(p.file)), manifest);
const meta = manifest.meta;
const tok = new core.BPETokenizer(data(manifest.tokenizer.file));
let t0 = performance.now();
const engine = new RillEngine(bytes, data(manifest.engine.file));
const tLoad = performance.now() - t0;
const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
const refAll = JSON.parse(readFileSync(refPath, "utf8"));
const cases = (refAll.cases || [...refAll.single, ...refAll.three]).slice(0, nMax ? +nMax : undefined);
let dz = 0, dh = 0, dpRef = 0, dpOrt = 0, agreeOrt = 0, agreeRef = 0, nq = 0; const tJs = [], tOrt = [];
for (const c of cases) {
  const rec = core.toRecord(c.request), pk = core.pack(tok, meta, rec, 1500), L = pk.ids.length;
  const i64 = (a) => BigInt64Array.from(a.map((v) => BigInt(v)));
  t0 = performance.now();
  const out = await sess.run({ input_ids: new ort.Tensor("int64", i64(pk.ids), [1, L]), segment_ids: new ort.Tensor("int64", i64(pk.seg), [1, L]),
                               position_ids: new ort.Tensor("int64", i64(pk.pos), [1, L]), slot_index: new ort.Tensor("int64", i64(pk.slots), [pk.slots.length]) });
  tOrt.push(performance.now() - t0);
  t0 = performance.now();
  const js = engine.run(pk.ids, pk.seg, pk.pos, pk.slots);
  tJs.push(performance.now() - t0);
  for (let k = 0; k < js.slot_logits.length; k++) dz = Math.max(dz, Math.abs(js.slot_logits[k] - out.slot_logits.data[k]));
  for (let k = 0; k < js.slot_hidden.length; k++) dh = Math.max(dh, Math.abs(js.slot_hidden[k] - out.slot_hidden.data[k]));
  const aJ = core.answers(core.calibrate(js.slot_logits, js.slot_hidden, pk.layout, meta.calibrator).logits, pk.layout);
  const aO = core.answers(core.calibrate(out.slot_logits.data, out.slot_hidden.data, pk.layout, meta.calibrator).logits, pk.layout);
  let i = 0;
  for (const q of pk.layout) {
    const pj = aJ[q.qid].p, po = aO[q.qid].p, pr = c.p.slice(i, i + q.K); i += q.K;
    for (let k = 0; k < q.K; k++) { dpOrt = Math.max(dpOrt, Math.abs(pj[k] - po[k])); dpRef = Math.max(dpRef, Math.abs(pj[k] - pr[k])); }
    const am = (v) => v.indexOf(Math.max(...v));
    agreeOrt += am(pj) === am(po); agreeRef += am(pj) === am(pr); nq++;
  }
}
const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log(JSON.stringify({ cases: cases.length, questions: nq, max_dlogit_vs_ort: dz, max_dhidden_vs_ort: dh, max_dp_vs_ort: dpOrt, max_dp_vs_torch: dpRef,
  argmax_agree_ort: agreeOrt / nq, argmax_agree_torch: agreeRef / nq, js_load_ms: Math.round(tLoad), js_median_ms: Math.round(med(tJs)), ort_median_ms: Math.round(med(tOrt)) }));
