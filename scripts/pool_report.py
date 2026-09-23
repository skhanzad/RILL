"""Post-training under task shift: compare Stage 1, Stage 1 + temperature scaling (fitted on the novel-task pool), supervised
fine-tuning on the pool, and RL variants on the pool. Metrics on decision-v7 dev, transfer-v4 dev and the pool's held-out
split, all computed by kev.metrics.

    python scripts/pool_report.py --tag 68B
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402,F401
from kev.benchmark import summarize  # noqa: E402
from kev.metrics import served, tempered_row  # noqa: E402

PARTS = [("decision-v7", "development"), ("transfer-v4", "development"), ("pool", "heldout")]
KEYS = ("acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def load(d, suite, split):
    p = Path(d) / f"{suite}.{split}.rows.json"
    return json.loads(p.read_text()) if p.exists() else None


def metrics_of(rows):
    return {k: summarize(rows, 1.0, ())["clean"].get(k) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="68B")
    ap.add_argument("--split", default="development", choices=["development", "test"])
    a = ap.parse_args()
    global PARTS
    root = "pool" if a.split == "development" else "pooltest"
    PARTS = [("decision-v7", a.split), ("transfer-v4", a.split), ("pool", "heldout")]
    base = ROOT / f"results/{root}/base-{a.tag}"
    variants = {"Stage 1": base, "SFT on pool": ROOT / f"results/{root}/sft-{a.tag}",
                "RL (GRPO + Brier term)": ROOT / f"results/{root}/rl-full-{a.tag}",
                "RL (Brier term only)": ROOT / f"results/{root}/rl-nopg-{a.tag}",
                "RL (Brier term only, beta 0.5)": ROOT / f"results/{root}/rl-nopg-b0.5-{a.tag}",
                "RL (GRPO only)": ROOT / f"results/{root}/rl-nocal-{a.tag}",
                "RL calibrator (GRPO + Brier term)": ROOT / f"results/{root}/calrl-full-{a.tag}",
                "RL calibrator (Brier term only)": ROOT / f"results/{root}/calrl-brier-{a.tag}",
                "RL calibrator (GRPO only)": ROOT / f"results/{root}/calrl-pg-{a.tag}",
                "RL calibrator, in-distribution pool": ROOT / f"results/{root}/calrl-cal-full-{a.tag}"}
    out = {}
    for name, d in variants.items():
        if not d.exists():
            continue
        out[name] = {f"{s}.{p}": metrics_of(load(d, s, p)) for s, p in PARTS if load(d, s, p) is not None}
        if name == "Stage 1":
            fit = load(d, "pool", "train")
            T, _ = served(fit, fit)
            out["Stage 1 + TS (pool)"] = {"T": T, **{f"{s}.{p}": metrics_of([tempered_row(r, T) for r in load(d, s, p)]) for s, p in PARTS}}
    (ROOT / f"results/{root}/report-{a.tag}.json").write_text(json.dumps(out, indent=1))
    for name, v in out.items():
        print(f"== {name}" + (f"  (T={v['T']:.2f})" if "T" in v else ""))
        for part in [f"{s}.{p}" for s, p in PARTS]:
            if part in v:
                print(f"   {part:28s}", {k: round(x, 4) if isinstance(x, float) else x for k, x in v[part].items()})


if __name__ == "__main__":
    main()
