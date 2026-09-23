"""Tables for the architecture/objective ablations (runs/abl68-*, 100k-request subset, 1 epoch) and the System-2 step
sweep (results/rill/rill-68m-B[-nfeN]). Development partitions only (model selection data), metrics from kev.metrics.

    python scripts/ablation_report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402,F401
from kev.benchmark import summarize  # noqa: E402

ABL = [("full model", "abl68-ref"), ("cross-entropy classifier (no flow, no self-cond.)", "abl68-ce-only"),
       ("no ConvNeXt V2 blocks", "abl68-no-convnext"), ("no self-conditioning", "abl68-no-selfcond"),
       ("one-hot score encoding (no CDF)", "abl68-onehot-score"), ("flow loss only (no cross-entropy)", "abl68-no-ce")]
KEYS = ("acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def main():
    out = {"ablations": {}, "nfe": {}}
    for name, run in ABL:
        f = ROOT / "runs" / run / "best.json"
        if f.exists():
            r = json.loads(f.read_text())
            out["ablations"][name] = {s: {k: r[s].get(k) for k in KEYS if k in r[s]} for s in ("decision-v7", "transfer-v4")}
    for nfe, d in ((1, "rill-68m-B"), (2, "rill-68m-B-nfe2"), (4, "rill-68m-B-nfe4"), (8, "rill-68m-B-nfe8")):
        res = {}
        for suite in ("decision-v7", "transfer-v4"):
            p = ROOT / "results/rill" / d / f"{suite}.development.rows.json"
            if p.exists():
                c = summarize(json.loads(p.read_text()), 1.0, ())["clean"]
                res[suite] = {k: c.get(k) for k in KEYS}
        if res:
            out["nfe"][nfe] = res
    # score-question metrics for the encoding ablation (ranked probability score on ordinal questions)
    (ROOT / "results/ablations.json").write_text(json.dumps(out, indent=1))
    for sec, v in out.items():
        print("==", sec)
        for name, r in v.items():
            print(f"  {str(name):52s}", {s: {k: round(x, 3) for k, x in m.items() if x is not None} for s, m in r.items()})


if __name__ == "__main__":
    main()
