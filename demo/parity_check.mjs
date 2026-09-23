// One-time parity check of rill-core.js (tokenizer + packing) against the Python reference layouts.
import { readFileSync } from "fs";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const core = require("./site/rill-core.js");
const [tokPath, refPath] = process.argv.slice(2);
const tok = new core.BPETokenizer(JSON.parse(readFileSync(tokPath, "utf8")));
const ref = JSON.parse(readFileSync(refPath, "utf8"));
let ok = 0, bad = 0;
for (const c of ref.cases) {
  const p = core.pack(tok, ref.meta, core.toRecord(c.request), 1500);
  const same = JSON.stringify(p.ids) === JSON.stringify(c.ids) && JSON.stringify(p.seg) === JSON.stringify(c.seg) &&
               JSON.stringify(p.pos) === JSON.stringify(c.pos) && JSON.stringify(p.slots) === JSON.stringify(c.slots.flat());
  if (same) ok++; else { bad++; if (bad <= 3) { const i = p.ids.findIndex((v, k) => v !== c.ids[k]); console.log("MISMATCH at", i, "js", p.ids.slice(Math.max(0,i-3), i+5), "py", c.ids.slice(Math.max(0,i-3), i+5)); } }
}
console.log(`parity: ${ok} identical, ${bad} different, of ${ref.cases.length}`);
