"""Score a Rill (DecisionFlow) checkpoint on every evaluation partition with kev's metric code, writing rows/report
files in the same layout as baselines/run.py.

    python scripts/eval_rill.py --ckpt runs/df68-B/best.pt --out results/rill/rill-68m
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402
import decisionflow.evaluate as ev  # noqa: E402
from decisionflow.evaluate import predict_records  # noqa: E402
from decisionflow.layout import Layout  # noqa: E402
from decisionflow.train import load  # noqa: E402
from kev.benchmark import prediction_rows, summarize  # noqa: E402
from kev.suite import load_split, read_jsonl, read_manifest  # noqa: E402

PARTS = [("decision-v7", "test"), ("transfer-v4", "test"), ("transfer-v9", "test"), ("semif-v1", "development"),
         ("wanli-v1", "development"), ("scienthoon-v1", "development"), ("decision-v7", "calibration"),
         ("decision-v7", "development"), ("transfer-v4", "development")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--noise", default="zero")
    ap.add_argument("--parts", default="all")
    ap.add_argument("--canonical", type=int, default=0, help="1: sort choice options by name before packing (order invariance)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    ev.CANONICAL_ORDER = bool(a.canonical)
    if a.device == "cpu":
        import torch
        torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = load(a.ckpt, a.device)
    if a.device == "cpu":
        model = model.float()
    lay = Layout(tok, max_state=7000)
    summary = {"ckpt": a.ckpt, "nfe": a.nfe, "noise": a.noise, "canonical": a.canonical, "config": cfg, "params": sum(p.numel() for p in model.parameters())}
    parts = PARTS if a.parts == "all" else [tuple(p.split(":")) for p in a.parts.split(",")]
    for suite, split in parts:
        if suite == "file":          # --parts file:data/rlpool/heldout.jsonl  (labelled requests with _meta)
            records = read_jsonl(split); suite, split = "pool", Path(split).stem
        else:
            records = load_split(SUITES[suite], split, allow_test=(split == "test"))
        t0 = time.time()
        preds = predict_records(model, lay, records, a.device, nfe=a.nfe, noise=a.noise)
        rows = []
        for r, pr in zip(records, preds):
            new = prediction_rows(r, pr)
            for row in new:
                row["unsupported"] = False
            rows += new
        report = summarize(rows, 1.0, tuple(read_manifest(SUITES[suite]).get("holdout_sources", [])) if suite in SUITES else ())
        report["coverage"] = {"questions": len(rows), "unsupported_questions": 0, "hard_failures": 0}
        (out / f"{suite}.{split}.rows.json").write_text(json.dumps(rows))
        (out / f"{suite}.{split}.report.json").write_text(json.dumps(report, indent=1))
        c = report["clean"]
        summary[f"{suite}.{split}"] = {k: c[k] for k in ("acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error")}
        print(suite, split, json.dumps(summary[f"{suite}.{split}"]), f"{time.time()-t0:.1f}s", flush=True)
    (out / f"summary_nfe{a.nfe}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
