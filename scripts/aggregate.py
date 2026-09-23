"""Aggregate saved prediction rows of every system into metrics, bootstrap intervals, temperature-scaled variants, LaTeX
tables and the numbers macro file used by the paper. All metrics are computed by kev.metrics / kev.benchmark.

    python scripts/aggregate.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402,F401  (kev import path)
from kev.benchmark import summarize  # noqa: E402
from kev.metrics import cluster_resamples, metrics, scored_rows, served, tempered_row, unknowable_report  # noqa: E402

RES = ROOT / "results"
MAIN = [("decision-v7", "test"), ("transfer-v4", "test"), ("transfer-v9", "test"), ("semif-v1", "development"),
        ("wanli-v1", "development"), ("scienthoon-v1", "development")]


def load_rows(d, suite, split):
    p = Path(d) / f"{suite}.{split}.rows.json"
    return json.loads(p.read_text()) if p.exists() else None


def ci(rows, stat, samples=1000, seed=0):
    rs = scored_rows(rows)
    vals = []
    for idx in cluster_resamples(rs, samples, seed):
        vals.append(stat([rs[i] for i in idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def acc_stat(rs):
    return float(np.mean([int(np.argmax(r["p"]) == r["label"]) for r in rs]))


def system_report(d, temperature_from_cal=True, boot=True, shipped_T=None):
    """{suite: {"raw": {...}, "ts": {...} | None, "T": float | None}} for one system directory."""
    out = {}
    cal = load_rows(d, "decision-v7", "calibration")
    T = None
    cal_l = [r for r in cal if "logits" in r] if cal is not None else []
    if temperature_from_cal and cal_l and len(scored_rows(cal_l)) > 0.9 * len(scored_rows(cal)):
        T, _ = served(cal_l, cal_l)
    for suite, split in MAIN:
        rows = load_rows(d, suite, split)
        if rows is None:
            continue
        rep = summarize(rows, 1.0, ())
        c = rep["clean"]
        entry = {"raw": {k: c.get(k) for k in ("n", "acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error",
                                               "aurc", "mean_conf", "score_mae", "ranked_probability_score")},
                 "variants": {k: v.get("acc") for k, v in rep["variants"].items()},
                 "perm_flip": rep["permutation"]["flip_rate"], "unknowable": rep["unknowable"], "paired_flip": rep["paired_flip"],
                 "unsupported": int(sum(1 for r in rows if r.get("unsupported"))), "n_rows": len(rows)}
        entry["tasks"] = {t: {"acc": v["acc"], "n": v["n"]} for t, v in rep["tasks"].items()}
        if boot:
            entry["acc_ci"] = ci(rows, acc_stat)
        if T is not None and all("logits" in r for r in rows if not r.get("unsupported")):
            trows = [tempered_row(r, T) if "logits" in r else r for r in rows]
            trep = summarize(trows, 1.0, ())
            tc = trep["clean"]
            entry["ts"] = {k: tc.get(k) for k in ("acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")}
            entry["ts_unknowable"] = trep["unknowable"]
            entry["ts_variants"] = {k: v.get("acc") for k, v in trep["variants"].items()}
        entry["T"] = T
        if shipped_T is not None:            # the system as released: its checkpoints ship a temperature that the loader applies
            srows = [tempered_row(r, shipped_T) if "logits" in r else r for r in rows]
            srep = summarize(srows, 1.0, ())
            sc = srep["clean"]
            entry["served"] = {k: sc.get(k) for k in ("n", "acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error",
                                                      "aurc", "mean_conf")}
            entry["served_unknowable"] = srep["unknowable"]
            entry["shipped_T"] = shipped_T
        out[suite] = entry
    return out


def main():
    registry = json.loads((RES / "registry.json").read_text())
    allrep = {}
    for sysname, info in registry.items():
        d = ROOT / info["dir"]
        if not d.exists():
            print("missing", sysname, d); continue
        allrep[sysname] = {"info": info, "suites": system_report(d, shipped_T=info.get("shipped_T"))}
        print(sysname, {s: round(v["raw"]["acc"], 3) for s, v in allrep[sysname]["suites"].items()}, flush=True)
    (RES / "aggregate.json").write_text(json.dumps(allrep, indent=1))


if __name__ == "__main__":
    main()
