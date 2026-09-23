// Pure-JavaScript System-1 readout for Rill: the fallback when a browser or host does not allow WebAssembly.
// It reads the 8-bit weights out of the demo's ONNX file (int8 embeddings, MatMulNBits blocks) and the small fp32 tensors
// from model/rill-js.json (norms, ConvNeXt vectors, and the flow-branch modulations evaluated once at t = 0), then runs
// decisionflow/model.py's text branch and flow branch for one packed request. demo/engine_check.mjs compares it with
// ONNX Runtime. Works in browsers (window.RillEngine) and in Node (module.exports).
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- minimal protobuf reading of an ONNX ModelProto
  function varint(b, p) {
    let x = 0, mul = 1, byte;
    do { byte = b[p++]; x += (byte & 0x7f) * mul; mul *= 128; } while (byte & 0x80);
    return [x, p];
  }

  // calls fn(field, wireType, a, b): varint -> (value, next position); length-delimited/fixed -> (start, end)
  function fields(b, start, end, fn) {
    let p = start;
    while (p < end) {
      let key; [key, p] = varint(b, p);
      const f = Math.floor(key / 8), wt = key & 7;
      if (wt === 0) { let v; [v, p] = varint(b, p); fn(f, wt, v, p); }
      else if (wt === 2) { let n; [n, p] = varint(b, p); fn(f, wt, p, p + n); p += n; }
      else if (wt === 5) { fn(f, wt, p, p + 4); p += 4; }
      else if (wt === 1) { fn(f, wt, p, p + 8); p += 8; }
      else throw new Error("unsupported protobuf wire type " + wt);
    }
  }

  const utf8 = new TextDecoder();

  // ModelProto.graph (7) -> GraphProto.initializer (5) -> TensorProto {dims 1, data_type 2, float_data 4, int32_data 5, name 8, raw_data 9}
  function initializers(bytes, wanted) {
    const out = new Map();
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    fields(bytes, 0, bytes.length, (f, wt, s, e) => {
      if (f !== 7 || wt !== 2) return;
      fields(bytes, s, e, (gf, gwt, gs, ge) => {
        if (gf !== 5 || gwt !== 2) return;
        const t = { name: null, dims: [], dtype: 0, raw: null, floats: [], ints: [] };
        fields(bytes, gs, ge, (tf, twt, a, b) => {
          if (tf === 8) t.name = utf8.decode(bytes.subarray(a, b));
          else if (tf === 2) t.dtype = a;
          else if (tf === 9) t.raw = bytes.subarray(a, b);
          else if (tf === 1 || tf === 5) {
            const dst = tf === 1 ? t.dims : t.ints;
            if (twt === 0) dst.push(a); else { let p = a; while (p < b) { let v; [v, p] = varint(bytes, p); dst.push(v); } }
          } else if (tf === 4) {
            if (twt === 5) t.floats.push(dv.getFloat32(a, true)); else for (let p = a; p < b; p += 4) t.floats.push(dv.getFloat32(p, true));
          }
        });
        if (t.name !== null && wanted.has(t.name)) out.set(t.name, t);
      });
    });
    for (const n of wanted) if (!out.has(n)) throw new Error("ONNX initializer not found: " + n);
    return out;
  }

  function f32(t) {                                           // FLOAT tensor -> Float32Array (copied, so aligned)
    if (t.raw) return new Float32Array(t.raw.slice().buffer);
    return Float32Array.from(t.floats);
  }

  function b64f32(s) {
    const bin = typeof atob === "function" ? atob(s) : Buffer.from(s, "base64").toString("binary");
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }

  // MatMulNBits, 8 bits, symmetric: B [N, K/block, block] uint8 with implicit zero point 128, one scale per block
  function dequant(tB, tS, N, K, block) {
    const B = tB.raw, S = f32(tS), nb = Math.ceil(K / block), W = new Float32Array(N * K);
    for (let n = 0; n < N; n++) {
      for (let b = 0; b < nb; b++) {
        const s = S[n * nb + b], base = (n * nb + b) * block, k0 = b * block;
        for (let j = 0; j < block && k0 + j < K; j++) W[n * K + k0 + j] = (B[base + j] - 128) * s;
      }
    }
    return W;
  }

  // ---------------------------------------------------------------- numerics
  function erf(x) {                                           // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));

  function layerNorm(x, L, d, w, b, eps) {
    const y = new Float32Array(L * d);
    for (let i = 0; i < L; i++) {
      const o = i * d;
      let m = 0; for (let j = 0; j < d; j++) m += x[o + j]; m /= d;
      let v = 0; for (let j = 0; j < d; j++) { const t = x[o + j] - m; v += t * t; } v /= d;
      const r = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < d; j++) y[o + j] = (x[o + j] - m) * r * w[j] + (b ? b[j] : 0);
    }
    return y;
  }

  // y[L, N] = x[L, K] W[N, K]^T + bias; four tokens share each weight row
  function linear(x, L, K, W, N, bias) {
    const y = new Float32Array(L * N);
    let i = 0;
    for (; i + 3 < L; i += 4) {
      const a0 = i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
      for (let n = 0; n < N; n++) {
        const wo = n * K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let k = 0; k < K; k++) {
          const w = W[wo + k];
          s0 += x[a0 + k] * w; s1 += x[a1 + k] * w; s2 += x[a2 + k] * w; s3 += x[a3 + k] * w;
        }
        const bb = bias ? bias[n] : 0;
        y[i * N + n] = s0 + bb; y[(i + 1) * N + n] = s1 + bb; y[(i + 2) * N + n] = s2 + bb; y[(i + 3) * N + n] = s3 + bb;
      }
    }
    for (; i < L; i++) {
      const a0 = i * K;
      for (let n = 0; n < N; n++) {
        const wo = n * K;
        let s = 0;
        for (let k = 0; k < K; k++) s += x[a0 + k] * W[wo + k];
        y[i * N + n] = s + (bias ? bias[n] : 0);
      }
    }
    return y;
  }

  // ---------------------------------------------------------------- the model
  class RillEngine {
    constructor(onnxBytes, spec) {
      const c = spec.config;
      this.c = c;
      const names = new Set([spec.embedding.q, spec.embedding.scale, spec.embedding.zero_point]);
      for (const m of Object.values(spec.matrices)) { names.add(m.B); names.add(m.scales); }
      const T = initializers(onnxBytes, names);
      const emb = T.get(spec.embedding.q);
      this.emb = emb.raw;                                      // [vocab, d] uint8
      this.embScale = f32(T.get(spec.embedding.scale))[0];
      const zp = T.get(spec.embedding.zero_point);
      this.embZp = zp.raw ? zp.raw[0] : zp.ints[0];
      this.W = {};
      for (const [k, m] of Object.entries(spec.matrices)) this.W[k] = dequant(T.get(m.B), T.get(m.scales), m.N, m.K, m.block);
      const s = spec.small;
      this.embNorm = b64f32(s.emb_norm); this.finalNorm = b64f32(s.final_norm);
      this.attnNorm = s.attn_norm.map((v) => (v ? b64f32(v) : null));
      this.mlpNorm = s.mlp_norm.map(b64f32);
      this.conv = s.convnext.map((b) => Object.fromEntries(Object.entries(b).map(([k, v]) => [k, b64f32(v)])));
      this.mods = s.mods.map(b64f32); this.finalMod = b64f32(s.final_mod); this.ans = b64f32(s.ans);
      this.headW = b64f32(s.head_w); this.headB = s.head_b;
      this.invFreq = Object.fromEntries(Object.entries(spec.rope_inv_freq).map(([k, v]) => [k, b64f32(v)]));
    }

    attention(x, L, l, seg, pos, rope) {
      const c = this.c, d = c.d, H = c.heads, D = d / H, half = D / 2;
      const local = c.layer_types[l] === "sliding_attention";
      const { cos, sin } = rope[c.layer_types[l]];
      const qkv = linear(x, L, d, this.W[`layers.${l}.Wqkv`], 3 * d, null);
      for (let i = 0; i < L; i++) {                             // rotary position embedding on q and k
        for (let part = 0; part < 2; part++) {
          for (let h = 0; h < H; h++) {
            const o = i * 3 * d + part * d + h * D;
            for (let j = 0; j < half; j++) {
              const cs = cos[i * half + j], sn = sin[i * half + j], a = qkv[o + j], b = qkv[o + j + half];
              qkv[o + j] = a * cs - b * sn;
              qkv[o + j + half] = b * cs + a * sn;
            }
          }
        }
      }
      const out = new Float32Array(L * d), sc = new Float64Array(L), scale = 1 / Math.sqrt(D);
      for (let i = 0; i < L; i++) {
        for (let h = 0; h < H; h++) {
          const qo = i * 3 * d + h * D;
          let mx = -Infinity;
          for (let j = 0; j < L; j++) {
            // state tokens read the state; question tokens read the state and their own question; every token reads itself
            let ok = i === j || seg[i] === seg[j] || (seg[j] === 0 && seg[i] > 0);
            if (ok && local && i !== j) ok = Math.abs(pos[i] - pos[j]) <= c.window;
            if (!ok) { sc[j] = -Infinity; continue; }
            const ko = j * 3 * d + d + h * D;
            let s = 0;
            for (let t = 0; t < D; t++) s += qkv[qo + t] * qkv[ko + t];
            sc[j] = s * scale;
            if (sc[j] > mx) mx = sc[j];
          }
          let z = 0;
          for (let j = 0; j < L; j++) { const e = sc[j] === -Infinity ? 0 : Math.exp(sc[j] - mx); sc[j] = e; z += e; }
          const oo = i * d + h * D;
          for (let j = 0; j < L; j++) {
            if (sc[j] === 0) continue;
            const w = sc[j] / z, vo = j * 3 * d + 2 * d + h * D;
            for (let t = 0; t < D; t++) out[oo + t] += w * qkv[vo + t];
          }
        }
      }
      return linear(out, L, d, this.W[`layers.${l}.Wo`], d, null);
    }

    mlp(x, L, l) {
      const d = this.c.d, I = this.c.inter;
      const u = linear(x, L, d, this.W[`layers.${l}.Wi`], 2 * I, null), g = new Float32Array(L * I);
      for (let i = 0; i < L; i++) for (let j = 0; j < I; j++) g[i * I + j] = gelu(u[i * 2 * I + j]) * u[i * 2 * I + I + j];
      return linear(g, L, I, this.W[`layers.${l}.Wo2`], d, null);
    }

    convnext(h, L, b, seg) {
      const c = this.c, d = c.d, d2 = 2 * d, P = this.conv[b], r = (c.kernel - 1) / 2, y = new Float32Array(L * d);
      for (let i = 0; i < L; i++) {
        for (let ch = 0; ch < d; ch++) {
          let s = h[i * d + ch] * P.dw[r * d + ch];
          for (let t = 1; t <= r; t++) {                         // the depthwise kernel never crosses a segment boundary
            if (i + t < L && seg[i + t] === seg[i]) s += h[(i + t) * d + ch] * P.dw[(r + t) * d + ch];
            if (i - t >= 0 && seg[i - t] === seg[i]) s += h[(i - t) * d + ch] * P.dw[(r - t) * d + ch];
          }
          y[i * d + ch] = s + P.dw_b[ch];
        }
      }
      const u = linear(layerNorm(y, L, d, P.norm_w, P.norm_b, c.convnext_eps), L, d, this.W[`convnext.${b}.pw1`], d2, P.pw1_b);
      for (let k = 0; k < u.length; k++) u[k] = gelu(u[k]);
      const sq = new Map();                                    // GRN with its L2 norm taken per segment
      for (let i = 0; i < L; i++) {
        if (!sq.has(seg[i])) sq.set(seg[i], new Float64Array(d2));
        const a = sq.get(seg[i]);
        for (let ch = 0; ch < d2; ch++) a[ch] += u[i * d2 + ch] * u[i * d2 + ch];
      }
      const nrm = new Map();
      for (const [k, a] of sq) {
        const g = a.map((v) => Math.sqrt(v + 1e-6));
        const mean = g.reduce((x, v) => x + v, 0) / d2;
        nrm.set(k, g.map((v) => v / (mean + 1e-6)));
      }
      for (let i = 0; i < L; i++) {
        const n = nrm.get(seg[i]);
        for (let ch = 0; ch < d2; ch++) { const v = u[i * d2 + ch]; u[i * d2 + ch] = P.gamma[ch] * (v * n[ch]) + P.beta[ch] + v; }
      }
      const o = linear(u, L, d2, this.W[`convnext.${b}.pw2`], d, P.pw2_b);
      for (let k = 0; k < h.length; k++) h[k] += o[k];
    }

    // one packed request -> raw option logits (S) and the slot representations (S x d) the calibrator reads
    run(ids, seg, pos, slots) {
      const c = this.c, d = c.d, L = ids.length, eps = c.eps;
      let h = new Float32Array(L * d);
      for (let i = 0; i < L; i++) {
        const row = ids[i] * d;
        for (let j = 0; j < d; j++) h[i * d + j] = (this.emb[row + j] - this.embZp) * this.embScale;
      }
      h = layerNorm(h, L, d, this.embNorm, null, eps);
      const rope = {};
      for (const [lt, inv] of Object.entries(this.invFreq)) {
        const half = inv.length, cos = new Float32Array(L * half), sin = new Float32Array(L * half);
        for (let i = 0; i < L; i++) for (let j = 0; j < half; j++) { const a = pos[i] * inv[j]; cos[i * half + j] = Math.cos(a); sin[i * half + j] = Math.sin(a); }
        rope[lt] = { cos, sin };
      }
      const add = (dst, src, gate, sq) => {                    // dst += (1 + gate) * src on question tokens, dst += src on the state
        for (let i = 0; i < L; i++) {
          const q = sq && seg[i] > 0;
          for (let j = 0; j < d; j++) dst[i * d + j] += (q ? 1 + gate[j] : 1) * src[i * d + j];
        }
      };
      const modulate = (x, scale, shift) => {
        for (let i = 0; i < L; i++) if (seg[i] > 0) for (let j = 0; j < d; j++) x[i * d + j] = x[i * d + j] * (1 + scale[j]) + shift[j];
        return x;
      };
      for (let l = 0; l < c.n_txt; l++) {                      // text branch
        const x = this.attnNorm[l] ? layerNorm(h, L, d, this.attnNorm[l], null, eps) : h;
        add(h, this.attention(x, L, l, seg, pos, rope), null, false);
        add(h, this.mlp(layerNorm(h, L, d, this.mlpNorm[l], null, eps), L, l), null, false);
      }
      for (let b = 0; b < this.conv.length; b++) this.convnext(h, L, b, seg);
      for (const s of slots) for (let j = 0; j < d; j++) h[s * d + j] += this.ans[j];   // answer track at t = 0
      for (let k = 0; k < c.n_dit; k++) {                      // DiT blocks, adaLN from t = 0 on question tokens
        const l = c.n_txt + k, M = this.mods[k];
        const sa = M.subarray(0, d), ba = M.subarray(d, 2 * d), ga = M.subarray(2 * d, 3 * d);
        const sm = M.subarray(3 * d, 4 * d), bm = M.subarray(4 * d, 5 * d), gm = M.subarray(5 * d, 6 * d);
        const x = modulate(layerNorm(h, L, d, this.attnNorm[l], null, eps), sa, ba);
        add(h, this.attention(x, L, l, seg, pos, rope), ga, true);
        const x2 = modulate(layerNorm(h, L, d, this.mlpNorm[l], null, eps), sm, bm);
        add(h, this.mlp(x2, L, l), gm, true);
      }
      const g = modulate(layerNorm(h, L, d, this.finalNorm, null, eps), this.finalMod.subarray(0, d), this.finalMod.subarray(d, 2 * d));
      const S = slots.length, z = new Float32Array(S), hid = new Float32Array(S * d);
      slots.forEach((s, k) => {
        let v = this.headB;
        for (let j = 0; j < d; j++) { v += this.headW[j] * g[s * d + j]; hid[k * d + j] = g[s * d + j]; }
        z[k] = v;
      });
      return { slot_logits: z, slot_hidden: hid };
    }
  }

  const api = { RillEngine, initializers };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RillEngine = RillEngine;
})(typeof self !== "undefined" ? self : this);
