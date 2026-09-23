"""Latency / memory benchmark on identical requests.

Requests: the first N records of transfer-v4 test (1 question each) and scienthoon-v1 (3 questions per state).
Systems: DecisionFlow checkpoints (PyTorch fp32 on CPU with 1/4 threads, dynamic-int8 on CPU, bf16 on GPU) and
baseline LLM readouts (SemIf direct letters / Kev) on GPU and, where feasible, CPU. Reports median ms per request.

    python scripts/bench_latency.py --system df --ckpt runs/df68-B/best.pt --device cpu --threads 4 --out results/latency/df68-cpu4.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baselines"))
import torch  # noqa: E402

from decisionflow.data import SUITES  # noqa: E402
from kev.suite import load_split  # noqa: E402


def requests(n):
    t4 = load_split(SUITES["transfer-v4"], "test", allow_test=True)
    sc = load_split(SUITES["scienthoon-v1"], "development")
    return {"single": t4[:n], "three": sc[:n]}


def time_calls(fn, recs, warmup=3, sync=None):
    for r in recs[:warmup]:
        fn(r)
    ts = []
    for r in recs:
        if sync: sync()
        t0 = time.perf_counter(); fn(r)
        if sync: sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(ts), "p90_ms": sorted(ts)[int(0.9 * len(ts)) - 1], "n": len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["df", "semif", "kev", "onnx"])
    ap.add_argument("--onnx", default="", help="onnx: exported graph (e.g. the 8-bit weight-only rill.g8w8.onnx)")
    ap.add_argument("--export_dir", default="", help="onnx: export directory with rill_meta.json (calibrator) and tokenizer/")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--int8", action="store_true", help="DecisionFlow: dynamic int8 quantisation of Linear layers (CPU)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    recs = requests(a.n)
    sync = (lambda: torch.cuda.synchronize()) if a.device == "cuda" else None
    info = vars(a).copy()
    if a.system == "df":
        from decisionflow.train import load
        from decisionflow.layout import Layout, collate
        from decisionflow.evaluate import request_to_rec
        model, tok, cfg = load(a.ckpt, "cpu")
        model = model.float().eval()
        if a.int8:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        if a.device == "cuda":
            model = model.cuda()
        lay = Layout(tok, max_state=7000)
        dt = torch.bfloat16 if (a.device == "cuda" and a.dtype == "bfloat16") else None

        @torch.no_grad()
        def fn(r):
            rec, meta = request_to_rec(r)
            b = collate([lay.encode(rec)], tok.pad_token_id).to(a.device)
            if dt is not None:
                with torch.autocast("cuda", dtype=dt):
                    return model.readout(b, nfe=1)
            return model.readout(b, nfe=1)
        info["params"] = sum(p.numel() for p in model.parameters()) if not a.int8 else None
    elif a.system == "onnx":
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from decisionflow.export import calibrate_np
        from decisionflow.layout import Layout, collate
        from decisionflow.evaluate import request_to_rec
        ed = Path(a.export_dir)
        cal = json.loads((ed / "rill_meta.json").read_text()).get("calibrator")
        tok = AutoTokenizer.from_pretrained(str(ed / "tokenizer"))
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
        lay = Layout(tok, max_state=7000)

        def fn(r):
            rec, meta = request_to_rec(r)
            b = collate([lay.encode(rec)], tok.pad_token_id)
            outs = sess.run(None, {"input_ids": b.ids.numpy(), "segment_ids": b.seg.numpy(), "position_ids": b.pos.numpy(),
                                   "slot_index": b.slot_i.numpy()})
            return calibrate_np(outs[0], outs[1], b.q_len.tolist(), cal)[0] if (cal and len(outs) > 1) else outs[0]
        info["params"] = None
        info["model_mb"] = Path(a.onnx).stat().st_size / 1e6
    else:
        from predictors import LetterScorer
        from run import per_question
        if a.system == "semif":
            pred = LetterScorer(a.model, a.revision, a.dtype) if a.device == "cuda" else None
            if pred is None:        # CPU: load in fp32/bf16 on CPU with the same prompt/readout
                import transformers
                from predictors import LetterScorer as LS
                pred = LS.__new__(LS)
                pred.tok = transformers.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
                cfgm = transformers.AutoConfig.from_pretrained(a.model, revision=a.revision)
                cls = transformers.AutoModelForCausalLM
                if cfgm.model_type in {"qwen3_5", "qwen3_5_text"}:
                    cls = transformers.Qwen3_5ForCausalLM; cfgm = cfgm.get_text_config()
                pred.model = cls.from_pretrained(a.model, revision=a.revision, config=cfgm, dtype=getattr(torch, a.dtype)).eval()
                pred.letter_prefix, pred.max_tokens, pred.fallback_rows, pred.temperature = "", 8192, 0, 1.0
                pred.meta = {}
            fn = lambda r: per_question(pred, r)
            info["params"] = sum(p.numel() for p in pred.model.parameters())
        else:
            import os
            from kev.checkpoint import LoadOptions
            from kev.predictors import LocalPredictor
            os.environ.setdefault("KEV_DTYPE", "fp32" if a.dtype == "float32" else "bf16")
            opts = LoadOptions.from_env(); opts = LoadOptions(**{**opts.__dict__, "temperature": 1.0})
            pred = LocalPredictor(a.model, a.device, opts, context={"max_state": 100000, "max_branch": 100000, "max_packed": 100000})
            fn = pred
            info["params"] = sum(p.numel() for p in pred.model.parameters())
    out = {"info": info}
    for k, rs in recs.items():
        out[k] = time_calls(fn, rs, sync=sync)
        print(k, out[k], flush=True)
    if a.device == "cuda":
        out["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
