"""Compare ONNX fp32 / int8 variants of the System-1 readout against the PyTorch model on a suite partition:
decision agreement, max |dp|, accuracy. CPU only.

    python scripts/check_quant.py --ckpt runs/df68-A/best.pt --onnx_dir export/test-68m-A --suite decision-v7 --split development
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

from decisionflow.data import SUITES  # noqa: E402
from decisionflow.export import calibrate_np  # noqa: E402
from decisionflow.evaluate import request_to_rec  # noqa: E402
from decisionflow.layout import Layout, collate  # noqa: E402
from decisionflow.train import load  # noqa: E402
from kev.suite import load_split  # noqa: E402


def softmax_groups(z, lens):
    out, i = [], 0
    for n in lens:
        e = np.exp(z[i:i + n] - z[i:i + n].max()); out.append(e / e.sum()); i += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--suite", default="decision-v7")
    ap.add_argument("--split", default="development")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variants", default="", help="comma-separated existing variant tags (rill_readout.<tag>.onnx); skips the dynamic-int8 sweep")
    ap.add_argument("--extra", default="", help="comma-separated tag=path.onnx of other graphs to compare (e.g. the demo's g8w8 model)")
    a = ap.parse_args()
    d = Path(a.onnx_dir)
    fp32 = d / "rill_readout.onnx"
    variants = {"fp32": fp32}
    if a.variants:
        for tag in a.variants.split(","):
            variants[tag] = d / f"rill_readout.{tag}.onnx"
    specs = {} if a.variants else {"int8_default": dict(weight_type=QuantType.QInt8),
             "int8_perchannel": dict(weight_type=QuantType.QInt8, per_channel=True),
             "int8_matmul_perchannel": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"]),
             "uint8_matmul_perchannel_rr": dict(weight_type=QuantType.QUInt8, per_channel=True, reduce_range=True, op_types_to_quantize=["MatMul"])}
    for name, kw in specs.items():
        p = d / f"rill_readout.{name}.onnx"
        if not p.exists():
            quantize_dynamic(str(fp32), str(p), **kw)
        variants[name] = p
    for item in [x for x in a.extra.split(",") if x]:
        tag, path = item.split("=", 1); variants[tag] = Path(path)
    meta = json.loads((d / "rill_meta.json").read_text()) if (d / "rill_meta.json").exists() else {}
    cal = meta.get("calibrator")
    model, tok, cfg = load(a.ckpt, "cpu"); model = model.float().eval()
    torch.set_num_threads(a.threads)
    lay = Layout(tok, max_state=7000)
    recs = load_split(SUITES[a.suite], a.split, allow_test=(a.split == "test"))[:a.limit]
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sessions = {k: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for k, p in variants.items()}
    stats = {k: {"agree": 0, "correct": 0, "maxdp": 0.0, "ms": []} for k in ["torch"] + list(variants)}
    n = 0
    for r in recs:
        rec, meta = request_to_rec(r)
        b = collate([lay.encode(rec)], tok.pad_token_id)
        lens = b.q_len.tolist()
        labels = []
        for qid, q in r["questions"].items():
            from kev.api import question_keys
            keys = question_keys(q["type"], q.get("criteria"))
            labels.append(keys.index(q["label"]) if q["type"] == "choice" else int(q["label"]))
        t0 = time.perf_counter()
        with torch.no_grad():
            zt = model.readout(b, nfe=1, return_logits=True)[1].numpy()
        stats["torch"]["ms"].append((time.perf_counter() - t0) * 1000)
        pt = softmax_groups(zt, lens)
        feeds = {"input_ids": b.ids.numpy(), "segment_ids": b.seg.numpy(), "position_ids": b.pos.numpy(), "slot_index": b.slot_i.numpy()}
        for k, s in sessions.items():
            t0 = time.perf_counter(); outs = s.run(None, feeds); stats[k]["ms"].append((time.perf_counter() - t0) * 1000)
            z = calibrate_np(outs[0], outs[1], lens, cal)[0] if (cal is not None and len(outs) > 1) else outs[0]
            pk = softmax_groups(z, lens)
            for p1, p2, y in zip(pk, pt, labels):
                stats[k]["agree"] += int(p1.argmax() == p2.argmax()); stats[k]["correct"] += int(p1.argmax() == y)
                stats[k]["maxdp"] = max(stats[k]["maxdp"], float(np.abs(p1 - p2).max()))
        for p2, y in zip(pt, labels):
            stats["torch"]["correct"] += int(p2.argmax() == y); stats["torch"]["agree"] += 1
        n += len(lens)
    out = {k: {"agreement": v["agree"] / n, "acc": v["correct"] / n, "max_abs_dp": v["maxdp"], "median_ms": float(np.median(v["ms"])),
               "size_mb": (variants[k].stat().st_size / 1e6 if k in variants else None)} for k, v in stats.items()}
    out["n_questions"] = n
    print(json.dumps(out, indent=1))
    (d / f"quant_check.{a.suite}.{a.split}{'.' + a.variants.replace(',', '_') if a.variants else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
