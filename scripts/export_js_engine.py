"""Constants for the pure-JavaScript Rill engine (demo/site/rill-engine.js), the fallback when a browser or host does not
allow WebAssembly. The engine reads the large matrices straight out of the demo's ONNX file (int8 embeddings and 8-bit
MatMulNBits weights), so nothing is shipped twice. This script (i) finds, by value, which ONNX initializer holds each
PyTorch weight (torch.onnx names transposed weights "onnx::MatMul_<n>"), and (ii) writes everything small in fp32:
norms, ConvNeXt vectors, the head, and the flow-branch modulations evaluated once at t = 0 (the System-1 readout never
uses another t).

    python scripts/export_js_engine.py --ckpt runs/rlcal32-C/best.pt --onnx demo/site_build/rill.g8w8.onnx --out demo/site/model

writes demo/site/model/engine.js (a data script, RillData.put("engine", ...)).
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.export import write_data_script  # noqa: E402
from decisionflow.train import load  # noqa: E402


def b64(t) -> str:
    a = (t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)).astype("<f4").ravel()
    return base64.b64encode(a.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok, cfg = load(a.ckpt, "cpu")
    model = model.float().eval()
    c = model.config
    d = model.d
    # ---------------------------------------------------------------- big matrices: ONNX MatMulNBits initializer <-> torch weight
    m = onnx.load(a.onnx)
    inits = {i.name: i for i in m.graph.initializer}
    wanted = {}
    for l, layer in enumerate(model.layers):
        wanted[f"layers.{l}.Wqkv"] = layer.attn.Wqkv.weight
        wanted[f"layers.{l}.Wo"] = layer.attn.Wo.weight
        wanted[f"layers.{l}.Wi"] = layer.mlp.Wi.weight
        wanted[f"layers.{l}.Wo2"] = layer.mlp.Wo.weight
    for b, blk in enumerate(model.convnext):
        wanted[f"convnext.{b}.pw1"] = blk.pw1.weight
        wanted[f"convnext.{b}.pw2"] = blk.pw2.weight
    cands = []
    for n in m.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        at = {x.name: x.i for x in n.attribute}
        assert at["bits"] == 8 and len(n.input) == 3, "expects symmetric 8-bit blocks without zero points"
        B = numpy_helper.to_array(inits[n.input[1]]).astype(np.int32) - 128          # [N, K/bs, bs]
        s = numpy_helper.to_array(inits[n.input[2]]).astype(np.float32).reshape(at["N"], -1)
        W = (B * s[:, :, None]).reshape(at["N"], -1)[:, :at["K"]]
        cands.append((n.input[1], n.input[2], at, W))
    mats, used = {}, set()
    for name, w in wanted.items():
        w = w.detach().numpy()
        ranked = sorted(((float(np.abs(W - w).max()), bn, sn, at) for bn, sn, at, W in cands if W.shape == w.shape and bn not in used),
                        key=lambda x: x[0])
        err, bn, sn, at = ranked[0]
        tol = float(np.abs(w).max()) / 127                                            # one quantisation step of the coarsest block
        runner_up = ranked[1][0] if len(ranked) > 1 else float("inf")
        assert err <= tol and err < 0.2 * runner_up, f"{name}: ambiguous or no match ({err:.3g}, tol {tol:.3g}, next {runner_up:.3g})"
        used.add(bn)
        mats[name] = {"B": bn, "scales": sn, "N": at["N"], "K": at["K"], "block": at["block_size"], "max_err": err}
    emb = {"q": "m.embeddings.tok_embeddings.weight_quantized", "scale": "m.embeddings.tok_embeddings.weight_scale",
           "zero_point": "m.embeddings.tok_embeddings.weight_zero_point"}
    for k in emb.values():
        assert k in inits, k
    # ---------------------------------------------------------------- small tensors and t = 0 constants
    with torch.no_grad():
        temb = F.silu(model.t_embed(torch.zeros(1)))
        mods = [mm(temb)[0] for mm in model.mods]                                        # (6d): sa, ba, ga, sm, bm, gm
        fmod = model.final_mod(temb)[0]                                                  # (2d): fs, fb
        ans = model.ans_in(torch.zeros(1, 3))[0]                                         # slot input for x_t = c = 0, flag 0
    small = {"emb_norm": b64(model.embeddings.norm.weight), "final_norm": b64(model.final_norm.weight),
             "attn_norm": [None if not hasattr(L.attn_norm, "weight") else b64(L.attn_norm.weight) for L in model.layers],
             "mlp_norm": [b64(L.mlp_norm.weight) for L in model.layers],
             "convnext": [{"dw": b64(blk.dw), "dw_b": b64(blk.dw_b), "norm_w": b64(blk.norm.weight), "norm_b": b64(blk.norm.bias),
                           "pw1_b": b64(blk.pw1.bias), "pw2_b": b64(blk.pw2.bias), "gamma": b64(blk.gamma), "beta": b64(blk.beta)}
                          for blk in model.convnext],
             "mods": [b64(x) for x in mods], "final_mod": b64(fmod), "ans": b64(ans),
             "head_w": b64(model.head.weight[0]), "head_b": float(model.head.bias[0])}
    for nm in ("embeddings.norm", "final_norm"):
        assert getattr(model.embeddings.norm if nm == "embeddings.norm" else model.final_norm, "bias") is None
    rope = {}
    for lt in set(model.layer_types):
        rope[lt] = b64(getattr(model.rotary, f"{lt}_inv_freq"))
        assert float(getattr(model.rotary, f"{lt}_attention_scaling", 1.0)) == 1.0
    conf = {"d": d, "heads": c.num_attention_heads, "inter": c.intermediate_size, "eps": c.norm_eps, "window": model.window,
            "layer_types": model.layer_types, "n_txt": model.n_txt, "n_dit": model.n_dit, "kernel": model.convnext[0].dw.shape[0],
            "convnext_eps": model.convnext[0].norm.eps, "activation": c.hidden_activation}
    assert c.hidden_activation == "gelu" and not c.mlp_bias and not c.attention_bias
    out = {"config": conf, "embedding": emb, "matrices": mats, "small": small, "rope_inv_freq": rope,
           "checkpoint": a.ckpt, "onnx": Path(a.onnx).name}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    size = write_data_script(Path(a.out) / "engine.js", "engine", out)       # loaded by the page as <script src>
    print(json.dumps({"matrices": len(mats), "max_err": max(v["max_err"] for v in mats.values()), "kB": size / 1e3}))


if __name__ == "__main__":
    main()
