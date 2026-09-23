"""Export the System-1 readout (one pass at t = 0) of a Rill checkpoint to ONNX (+ dynamic int8), for CPU/mobile/browser
inference with ONNX Runtime. Batch size 1; the graph takes the packed request and returns one logit per option slot;
the per-question softmax is applied by the caller. If the checkpoint carries an RL calibrator, the graph also returns the
slot representations and the calibrator weights go to rill_meta.json: the caller applies tau(x) per question
(calibrate_np below, and the same few lines in demo/site/rill-core.js), which needs the question grouping the graph
does not see.

    python -m decisionflow.export --ckpt runs/df32-B/best.pt --out export/rill-32m
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .layout import Batch, Layout, collate
from .train import load


def write_data_script(path, key: str, value) -> int:
    """Write  RillData.put("<key>", <JSON>);  -- a data file the browser demo loads as an ordinary script (no fetch, so
    no CORS from a sandboxed page), and whose payload stays plain JSON for Node tools (rill-core.js readDataScript).
    ASCII-only JSON keeps it valid in any JavaScript engine. Returns the file size in bytes."""
    text = 'RillData.put("' + key + '",' + json.dumps(value, separators=(",", ":"), ensure_ascii=True) + ");\n"
    Path(path).write_text(text, encoding="ascii")
    return len(text)


def calibrator_weights(cal) -> dict:
    """Calibrator parameters as nested lists (JSON), rounded to 7 significant digits."""
    r = lambda t: [float(f"{v:.7g}") for v in t.detach().float().reshape(-1).tolist()]
    return {"d": cal.h.in_features, "hidden": cal.h.out_features,
            "h_w": r(cal.h.weight), "h_b": r(cal.h.bias), "f_w": r(cal.f.weight), "f_b": r(cal.f.bias),
            "o_w": r(cal.out.weight), "o_b": r(cal.out.bias)}


def calibrate_np(z: np.ndarray, g: np.ndarray, lens, cal: dict):
    """Apply the RL calibrator outside the graph: per question, tau = softplus(o(silu(h(mean slot g) + f(stats)))) + 0.05
    and logits / tau. Mirrors DecisionFlow.calibrate / question_stats. Returns (calibrated logits, taus)."""
    H, d = cal["hidden"], cal["d"]
    hw, hb = np.array(cal["h_w"]).reshape(H, d), np.array(cal["h_b"])
    fw, fb = np.array(cal["f_w"]).reshape(H, 4), np.array(cal["f_b"])
    ow, ob = np.array(cal["o_w"]).reshape(H), float(cal["o_b"][0])
    out, taus, i = np.empty_like(z, dtype=np.float64), [], 0
    for n in lens:
        zq = z[i:i + n].astype(np.float64)
        lp = zq - zq.max() - np.log(np.exp(zq - zq.max()).sum()); p = np.exp(lp)
        rest = zq[zq < zq.max()]                                            # as question_stats: ties with the top are excluded
        stats = np.array([zq.max() - rest.max() if rest.size else 0.0, -(p * lp).sum(), np.log(n), p.max()])
        u = hw @ g[i:i + n].astype(np.float64).mean(0) + hb + fw @ stats + fb
        u = u / (1 + np.exp(-u))                                            # SiLU
        v = ow @ u + ob
        tau = (np.log1p(np.exp(-abs(v))) + max(v, 0.0)) + 0.05              # softplus + floor
        out[i:i + n] = zq / tau; taus.append(tau); i += n
    return out.astype(np.float32), taus


class Readout(nn.Module):
    """t = 0 readout for one packed request. Without a calibrator it returns the option logits; with one, the raw
    (uncalibrated) logits and the slot representations the calibrator reads."""

    def __init__(self, model, max_segments: int = 64):
        super().__init__()
        self.m = model
        self.max_segments = max_segments
        self.with_hidden = model.calibrator is not None

    def forward(self, ids, seg, pos, slot_i):
        L = ids.shape[1]
        S = slot_i.shape[0]
        valid = torch.ones_like(ids, dtype=torch.bool)
        slot_b = torch.zeros_like(slot_i)
        z0 = torch.zeros(S, dtype=torch.float32)
        b = Batch(ids, seg, pos, valid, slot_b, slot_i, slot_b, slot_b[:1], slot_b[:1], slot_b[:1],
                  torch.zeros(1, dtype=torch.bool), None, None, [])
        cache = self.m.text_branch(b, nseg=self.max_segments)
        if not self.with_hidden:
            return self.m.flow_branch(cache, b, z0, z0, torch.zeros(1), sc_flag=0.0)
        g = self.m.flow_branch(cache, b, z0, z0, torch.zeros(1), sc_flag=0.0, return_hidden=True)[0, slot_i]
        return self.m.head(g).squeeze(-1), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = load(a.ckpt, "cpu")
    model = model.float().eval()
    lay = Layout(tok, max_state=None)
    rec = {"state": "I was charged twice for order 4411. Please refund one charge today.",
           "questions": [{"instr": "Which team should handle this?", "options": ["billing: charges and refunds", "shipping", "returns"], "qtype": "choice"},
                         {"instr": "Is the customer angry?", "options": ["no", "yes"], "qtype": "noul"}]}
    enc = lay.encode(rec)
    b = collate([enc], tok.pad_token_id)
    wrapper = Readout(model).eval()
    cal = calibrator_weights(model.calibrator) if wrapper.with_hidden else None
    lens = b.q_len.tolist()
    args = (b.ids, b.seg, b.pos, b.slot_i)

    def finish(outs):
        """graph outputs -> final (calibrated if applicable) logits, as numpy"""
        if cal is None:
            return np.asarray(outs[0] if isinstance(outs, (list, tuple)) else outs)
        return calibrate_np(np.asarray(outs[0]), np.asarray(outs[1]), lens, cal)[0]

    with torch.no_grad():
        ref = wrapper(*args)
        ref_np = finish([t.numpy() for t in ref] if isinstance(ref, tuple) else ref.numpy())
        full = model.readout(b, nfe=1, return_logits=True)[1].numpy()
    print("wrapper (+calibrator) vs model readout max|diff|:", float(np.abs(ref_np - full).max()))
    onnx_path = out / "rill_readout.onnx"
    names = ["slot_logits", "slot_hidden"] if cal is not None else ["slot_logits"]
    axes = {"input_ids": {1: "L"}, "segment_ids": {1: "L"}, "position_ids": {1: "L"}, "slot_index": {0: "S"}, "slot_logits": {0: "S"}}
    if cal is not None:
        axes["slot_hidden"] = {0: "S"}
    torch.onnx.export(wrapper, args, str(onnx_path), input_names=["input_ids", "segment_ids", "position_ids", "slot_index"],
                      output_names=names, opset_version=a.opset, dynamo=False, dynamic_axes=axes)
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = out / "rill_readout.int8.onnx"
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    feeds = {"input_ids": b.ids.numpy(), "segment_ids": b.seg.numpy(), "position_ids": b.pos.numpy(), "slot_index": b.slot_i.numpy()}
    for p in (onnx_path, q_path):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        z = finish(sess.run(None, feeds))
        print(p.name, f"{p.stat().st_size/1e6:.1f} MB", "max|diff| vs torch:", float(np.abs(z - full).max()))
    tok.save_pretrained(out / "tokenizer")
    meta = {"q_token": "[unused0]", "opt_token": "[unused1]", "cls": tok.cls_token_id, "sep": tok.sep_token_id,
            "q_id": lay.q_id, "opt_id": lay.opt_id, "max_option_tokens": 64, "max_instr_tokens": 256, "config": cfg,
            "outputs": names, "calibrator": cal}
    (out / "rill_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
