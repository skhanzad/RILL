"""Score a baseline on frozen suites with kev's metric code.

    python baselines/run.py --system semif --model Qwen/Qwen3.5-4B --revision 851bf6e... --splits test --out results/semif-qwen35-4b
    python baselines/run.py --system kev --model jaredpalmer/kev-0.8b --out results/kev-0.8b
    python baselines/run.py --system nli --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --out results/nli-large
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "baselines"))
from decisionflow.data import SUITES  # noqa: E402  (kev path)
from kev.benchmark import prediction_rows, summarize  # noqa: E402
from kev.suite import load_split, read_manifest  # noqa: E402
from kev.api import question_keys  # noqa: E402

from predictors import Generative, LetterScorer, NLIZeroShot, Unsupported  # noqa: E402

DEFAULT_SUITES = {"decision-v7": "test", "transfer-v4": "test", "transfer-v9": "test",
                  "semif-v1": "development", "wanli-v1": "development", "scienthoon-v1": "development"}


def uniform_prediction(record):
    probs = {}
    for qid, q in record["questions"].items():
        keys = question_keys(q["type"], q.get("criteria"))
        probs[qid] = {k: 1.0 / len(keys) for k in keys}
    return {"probabilities": probs, "latency_ms": 0.0}


def per_question(predictor, record):
    """Call the predictor question by question so that one unsupported question does not drop the record."""
    out = {"probabilities": {}, "latency_ms": 0.0}
    has_logits = True
    unsupported = []
    for qid, q in record["questions"].items():
        sub = {**record, "questions": {qid: q}}
        try:
            pr = predictor(sub)
        except Unsupported:
            unsupported.append(qid)
            keys = question_keys(q["type"], q.get("criteria"))
            pr = {"probabilities": {qid: {k: 1.0 / len(keys) for k in keys}}, "latency_ms": 0.0}
            has_logits = False
        out["probabilities"][qid] = pr["probabilities"][qid]
        if "logits" in pr:
            out.setdefault("logits", {})[qid] = pr["logits"][qid]
        else:
            has_logits = False
        out["latency_ms"] += pr.get("latency_ms", 0.0)
    if not has_logits:
        out.pop("logits", None)
    else:
        out["inference_temperature"] = 1.0
    return out, unsupported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["semif", "letters", "generative", "nli", "kev"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant", default=None, choices=[None, "nf4", "int8"])
    ap.add_argument("--chat_template", default=None, help="path to a jinja chat template for models without one")
    ap.add_argument("--letter_prefix", default="", help="text between the generation prompt and the answer letter (' ' for 'ASSISTANT:' formats)")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    ap.add_argument("--splits", default="default", help="'default' (test for locked suites, development for external), or a split name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmpl = Path(a.chat_template).read_text() if a.chat_template else None
    t_load = time.time()
    if a.system in ("semif", "letters"):
        pred = LetterScorer(a.model, a.revision, a.dtype, chat_template=tmpl, quant=a.quant, letter_prefix=a.letter_prefix)
    elif a.system == "generative":
        pred = Generative(a.model, a.revision, a.dtype, chat_template=tmpl, quant=a.quant, letter_prefix=a.letter_prefix)
    elif a.system == "nli":
        pred = NLIZeroShot(a.model)
    else:
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        import os
        os.environ.setdefault("KEV_DTYPE", "fp32")
        opts = LoadOptions.from_env()
        opts = LoadOptions(**{**opts.__dict__, "temperature": 1.0})     # raw logits; any temperature is applied post hoc
        pred = LocalPredictor(a.model, "cuda", opts, context={"max_state": 100000, "max_branch": 100000, "max_packed": 100000})
    load_s = time.time() - t_load
    summary = {"system": a.system, "model": a.model, "revision": a.revision, "dtype": a.dtype, "quant": a.quant, "load_s": load_s}
    for suite in a.suites.split(","):
        split = DEFAULT_SUITES[suite] if a.splits == "default" else a.splits
        records = load_split(SUITES[suite], split, allow_test=(split == "test"))
        if a.limit:
            records = records[:a.limit]
        manifest = read_manifest(SUITES[suite])
        rows, unsupported, lat, fails = [], [], [], 0
        t0 = time.time()
        with (out / f"{suite}.{split}.predictions.jsonl").open("w") as f:
            for n, r in enumerate(records):
                try:
                    pr, uns = per_question(pred, r)
                except Exception as e:           # a hard failure is recorded and counted, never silently dropped
                    fails += 1
                    pr, uns = uniform_prediction(r), list(r["questions"])
                    print(f"FAIL {r['_meta']['id']}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                new = prediction_rows(r, pr)
                for row in new:
                    row["unsupported"] = row["question"] in uns
                rows += new; unsupported += [(r["_meta"]["id"], q) for q in uns]
                lat.append(pr["latency_ms"])
                f.write(json.dumps({"id": r["_meta"]["id"], "prediction": pr, "unsupported": uns}) + "\n")
                if (n + 1) % 200 == 0:
                    print(f"{suite} {n+1}/{len(records)} {time.time()-t0:.0f}s", flush=True)
        report = summarize(rows, 1.0, tuple(manifest.get("holdout_sources", [])))
        supported = [r for r in rows if not r["unsupported"]]
        report["supported_only"] = summarize(supported, 1.0, tuple(manifest.get("holdout_sources", []))) if supported and len(supported) < len(rows) else None
        report["coverage"] = {"questions": len(rows), "unsupported_questions": len(unsupported), "hard_failures": fails}
        report["latency_ms"] = {"median": sorted(lat)[len(lat) // 2], "mean": sum(lat) / len(lat)}
        if isinstance(pred, Generative):
            report["generative"] = {"total": pred.total, "invalid": pred.invalid}
            (out / f"{suite}.{split}.generations.json").write_text(json.dumps(pred.outputs[-len(rows):]))
            pred.total = pred.invalid = 0; pred.outputs = []
        if isinstance(pred, LetterScorer):
            report["tokenizer_fallback_rows"] = pred.fallback_rows; pred.fallback_rows = 0
        (out / f"{suite}.{split}.rows.json").write_text(json.dumps(rows))
        (out / f"{suite}.{split}.report.json").write_text(json.dumps(report, indent=1))
        c = report["clean"]
        summary[suite] = {"split": split, "acc": c["acc"], "nll": c["nll"], "brier": c["brier"], "ece": c["ece"],
                          "cer": c["confident_error_rate"], "cov5": c["coverage_at_5pct_error"], "unsupported": len(unsupported),
                          "fails": fails, "latency_ms": report["latency_ms"]["median"], "time_s": time.time() - t0}
        print(json.dumps({suite: summary[suite]}), flush=True)
    (out / f"summary_{a.splits}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
