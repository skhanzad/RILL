// Rill in the browser: tokenizer, request packing and answer decoding. Mirrors decisionflow/layout.py and kev/api.py
// exactly, so the browser returns the same probabilities as the Python model (checked by demo/parity_check.mjs).
// Works in browsers (window.RillCore) and in Node (module.exports) without dependencies.
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- byte-level BPE (GPT-NeoX / ModernBERT tokenizer)
  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const map = new Array(256);
    bs.forEach((b, i) => { map[b] = String.fromCodePoint(cs[i]); });
    return map;
  }

  const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class BPETokenizer {
    constructor(tok) {
      const m = tok.model;
      this.vocab = new Map(Object.entries(m.vocab));
      this.ranks = new Map();
      m.merges.forEach((mg, i) => {
        const pair = Array.isArray(mg) ? mg[0] + " " + mg[1] : mg;
        this.ranks.set(pair, i);
      });
      this.byteMap = bytesToUnicode();
      this.encoder = new TextEncoder();
      this.cache = new Map();
      // added tokens: leftmost-longest match before pre-tokenization (non-special ones are matched on normalised text)
      this.added = new Map();
      for (const a of tok.added_tokens) {
        const first = a.content[0];
        if (!this.added.has(first)) this.added.set(first, []);
        this.added.get(first).push({ s: a.content, id: a.id });
      }
      for (const list of this.added.values()) list.sort((x, y) => y.s.length - x.s.length);
      this.unused = (k) => this.idOf(`[unused${k}]`);
    }

    idOf(tokenString) {
      if (this.vocab.has(tokenString)) return this.vocab.get(tokenString);
      for (const list of this.added.values()) for (const a of list) if (a.s === tokenString) return a.id;
      return undefined;
    }

    bpe(word) {
      if (this.cache.has(word)) return this.cache.get(word);
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const merged = parts[best] + parts[best + 1];
        const next = [];
        for (let i = 0; i < parts.length; i++) {
          if (i < parts.length - 1 && parts[i] + " " + parts[i + 1] === parts[best] + " " + parts[best + 1]) {
            next.push(merged); i++;
          } else next.push(parts[i]);
        }
        parts = next;
      }
      const ids = parts.map((p) => this.vocab.get(p));
      if (this.cache.size < 50000) this.cache.set(word, ids);
      return ids;
    }

    encodePlain(text, out) {
      if (!text) return;
      for (const m of text.matchAll(PRETOKENIZE)) {
        const bytes = this.encoder.encode(m[0]);
        let s = "";
        for (const b of bytes) s += this.byteMap[b];
        for (const id of this.bpe(s)) out.push(id);
      }
    }

    encode(text) {
      text = (text || "").normalize("NFC");
      const out = [];
      let start = 0, i = 0;
      while (i < text.length) {
        const cands = this.added.get(text[i]);
        let hit = null;
        if (cands) for (const a of cands) if (text.startsWith(a.s, i)) { hit = a; break; }
        if (hit) {
          this.encodePlain(text.slice(start, i), out);
          out.push(hit.id);
          i += hit.s.length; start = i;
        } else i++;
      }
      this.encodePlain(text.slice(start), out);
      return out;
    }
  }

  // ---------------------------------------------------------------- request rendering (kev.api) and packing (layout.py)
  function pyStr(v) {
    if (v === true) return "True";
    if (v === false) return "False";
    if (typeof v === "number") return Number.isInteger(v) && !Object.is(v, -0) && String(v).indexOf("e") < 0 ? String(v) : String(v);
    return String(v);
  }

  function render(v, indent = 0) {
    const pad = "  ".repeat(indent);
    if (v === null || v === undefined) return "";
    if (typeof v !== "object") return pyStr(v);
    if (Array.isArray(v)) return v.map((x) => `${pad}- ${render(x, indent + 1).replace(/^\s+/, "")}`).join("\n");
    return Object.entries(v).map(([k, x]) => (x !== null && typeof x === "object")
      ? `${pad}${k}:\n${render(x, indent + 1)}` : `${pad}${k}: ${render(x)}`).join("\n");
  }

  function optionText(name, desc) {
    return (desc === null || desc === undefined || desc === "") ? name : `${name}: ${render(desc)}`;
  }

  // request = {state, questions: {qid: {type, instructions, criteria}}}  (the System One request shape)
  function toRecord(request) {
    const questions = [];
    for (const [qid, q] of Object.entries(request.questions)) {
      let keys, options, legend = null;
      if (q.type === "noul") {
        const c = q.criteria || {};
        keys = ["false", "true"];
        options = [optionText("no", c.false), optionText("yes", c.true)];
      } else if (q.type === "choice") {
        keys = Object.keys(q.criteria);
        options = keys.map((k) => optionText(k, q.criteria[k]));
      } else {
        options = q.criteria.map((x) => render(x));
        keys = options.map((_, i) => String(i));
        legend = Object.fromEntries(keys.map((k, i) => [k, options[i]]));
      }
      questions.push({ qid, type: q.type, instr: render(q.instructions), options, keys, legend });
    }
    return { state: render(request.state), questions };
  }

  function pack(tokenizer, meta, rec, maxState = 1500) {
    const CLS = meta.cls, SEP = meta.sep, Q = meta.q_id, OPT = meta.opt_id;
    const st = tokenizer.encode(rec.state).slice(0, maxState);
    const ids = [CLS, ...st, SEP];
    const S = ids.length;
    const seg = new Array(S).fill(0), pos = Array.from({ length: S }, (_, i) => i);
    const slots = [], layout = [];
    rec.questions.forEach((q, j) => {
      const qids = [Q, ...tokenizer.encode(q.instr).slice(0, meta.max_instr_tokens)];
      const start = slots.length;
      for (const opt of q.options) {
        slots.push(ids.length + qids.length);
        qids.push(OPT, ...tokenizer.encode(" " + opt).slice(0, meta.max_option_tokens));
      }
      qids.push(SEP);
      for (let k = 0; k < qids.length; k++) { ids.push(qids[k]); seg.push(j + 1); pos.push(S + k); }
      layout.push({ ...q, start, K: q.options.length });
    });
    return { ids, seg, pos, slots, layout };
  }

  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // RL calibrator: an input-dependent temperature per question, tau = softplus(o . silu(H hq + F stats)) + 0.05, where hq is
  // the mean slot representation and stats = (top-1 minus top-2 logit, entropy, log K, max p). Dividing by tau never changes
  // the arg-max. Mirrors DecisionFlow.calibrate and decisionflow/export.py calibrate_np.
  function calibrate(logits, hidden, layout, cal) {
    if (!cal || !hidden) return { logits: Float64Array.from(logits), taus: layout.map(() => 1) };
    const H = cal.hidden, d = cal.d, out = new Float64Array(logits.length), taus = [];
    for (const q of layout) {
      const n = q.K, z = Array.from(logits.slice(q.start, q.start + n), Number);
      const zmax = Math.max(...z);
      const lse = zmax + Math.log(z.reduce((a, v) => a + Math.exp(v - zmax), 0));
      const lp = z.map((v) => v - lse), p = lp.map(Math.exp);
      const rest = z.filter((v) => v < zmax);
      const stats = [rest.length ? zmax - Math.max(...rest) : 0, -p.reduce((a, v, i) => a + v * lp[i], 0), Math.log(n), Math.max(...p)];
      const hq = new Float64Array(d);
      for (let k = 0; k < n; k++) { const row = (q.start + k) * d; for (let i = 0; i < d; i++) hq[i] += hidden[row + i]; }
      for (let i = 0; i < d; i++) hq[i] /= n;
      let v = cal.o_b[0];
      for (let h = 0; h < H; h++) {
        let u = cal.h_b[h] + cal.f_b[h];
        const hr = h * d;
        for (let i = 0; i < d; i++) u += cal.h_w[hr + i] * hq[i];
        for (let i = 0; i < 4; i++) u += cal.f_w[h * 4 + i] * stats[i];
        v += cal.o_w[h] * (u / (1 + Math.exp(-u)));
      }
      const tau = Math.log1p(Math.exp(-Math.abs(v))) + Math.max(v, 0) + 0.05;
      for (let k = 0; k < n; k++) out[q.start + k] = z[k] / tau;
      taus.push(tau);
    }
    return { logits: out, taus };
  }

  // typed answers in the System One shape (kev.api.to_answers)
  function answers(logits, layout, taus) {
    const out = {};
    for (const q of layout) {
      const z = Array.from(logits.slice(q.start, q.start + q.K));
      const p = softmax(z);
      const K = p.length, top = p.indexOf(Math.max(...p));
      if (q.type === "noul") out[q.qid] = { type: "noul", noul: p[1], probabilities: { false: p[0], true: p[1] } };
      else if (q.type === "choice") out[q.qid] = { type: "choice", choice: q.keys[top], confidence: K > 1 ? (p[top] - 1 / K) / (1 - 1 / K) : 1,
        probabilities: Object.fromEntries(q.keys.map((k, i) => [k, p[i]])) };
      else {
        const score = p.reduce((a, v, i) => a + i * v, 0);
        const conf = K > 1 ? 1 - p.reduce((a, v, i) => a + v * Math.abs(i - top), 0) / (K - 1) : 1;
        out[q.qid] = { type: "score", score, confidence: conf, legend: q.legend, probabilities: Object.fromEntries(q.keys.map((k, i) => [k, p[i]])) };
      }
      out[q.qid].pmax = p[top];
      out[q.qid].top = top;
      out[q.qid].p = p;
      out[q.qid].tau = taus ? taus[layout.indexOf(q)] : 1;
    }
    return out;
  }

  // model parts (base64 text of consecutive gzip slices) -> the ONNX bytes
  function b64bytes(text) {
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(text);
    const bin = typeof atob === "function" ? atob(text) : Buffer.from(text, "base64").toString("binary");
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }

  async function decodeModel(texts, manifest) {
    const chunks = texts.map(b64bytes);
    const gz = new Uint8Array(chunks.reduce((a, c) => a + c.length, 0));
    let o = 0; for (const c of chunks) { gz.set(c, o); o += c.length; }
    if (manifest.encoding !== "gzip+base64") return gz;
    const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"));
    const out = new Uint8Array(await new Response(stream).arrayBuffer());
    if (out.length !== manifest.bytes) throw new Error(`model size ${out.length} != ${manifest.bytes}`);
    return out;
  }

  const api = { BPETokenizer, render, toRecord, pack, softmax, calibrate, answers, decodeModel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RillCore = api;
})(typeof self !== "undefined" ? self : this);
