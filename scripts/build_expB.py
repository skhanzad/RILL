"""Experiment B training pool: kev's ten trainable public sources at (up to) full train-split size, plus more records
from kev's own synthetic generators (trainable policy families, random rule structures with every held-out and locked
structure excluded), plus kev's released delta data. Every record whose normalised state hash (or semantic hash, for
synthetic cases) appears in ANY development / test / calibration partition of the evaluation suites is dropped.

    python scripts/build_expB.py --out data/expB
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import NIGHT2, SUITES  # noqa: E402  (sets up the kev path)
from kev.composition import generate as compose, sample_trees, canonical  # noqa: E402
from kev.data import ALL_REPOS, ALL_SOURCES, EVAL_ONLY, build  # noqa: E402
from kev.study_v3 import legacy  # noqa: E402
from kev.suite import load_split, read_jsonl, read_manifest, semantic_hash  # noqa: E402

PUBLIC = {"mnli": 120000, "boolq": 20000, "agnews": 40000, "dbpedia14": 40000, "yelp": 40000, "amazon": 40000,
          "imdb": 25000, "sst5": 20000, "trec": 20000, "banking77": 20000}
LEGACY_FAMILIES = ["return_window", "spend_threshold", "age_eligibility", "quantity_limit", "warranty_claim", "sla_response",
                   "late_fee", "volume_discount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="expB-20260923")
    ap.add_argument("--legacy_pairs", type=int, default=512, help="pairs per trainable policy family")
    ap.add_argument("--structures", type=int, default=240)
    ap.add_argument("--groups_per_structure", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 1) reserved states: every evaluation / calibration partition we will ever score or calibrate on
    reserved_text, reserved_sem = set(), set()
    for name, d in SUITES.items():
        for split in ("calibration", "development", "test"):
            try:
                recs = load_split(d, split, allow_test=True)
            except Exception:
                continue
            for r in recs:
                if "text_sha256" in r["_meta"]:
                    reserved_text.add(r["_meta"]["text_sha256"])
                reserved_sem.add(semantic_hash(r))
    print(f"reserved: {len(reserved_text)} state hashes, {len(reserved_sem)} semantic hashes", flush=True)
    v7 = read_manifest(SUITES["decision-v7"])
    assert not set(PUBLIC) & set(EVAL_ONLY)

    records, report = [], collections.Counter()
    # 2) public sources, drawn from their train splits with kev's converters
    for name, n in PUBLIC.items():
        recs = build(n, "train", seed=a.seed, only=(name,), sources={name: ALL_SOURCES[name]}, repos={name: ALL_REPOS[name]},
                     revisions=v7["dataset_revisions"])
        kept = [r for r in recs if r["_meta"]["text_sha256"] not in reserved_text and semantic_hash(r) not in reserved_sem]
        report[f"{name}_built"] += len(recs); report[f"{name}_dropped_reserved"] += len(recs) - len(kept)
        for r in kept:
            r["_meta"]["variant"] = "clean"; r["_meta"].setdefault("group_id", r["_meta"]["id"])
        records += kept
        print(f"{name}: built {len(recs)} kept {len(kept)}", flush=True)
    # also keep decision-v7's own training records (a superset of Experiment A's public data)
    v7_train = load_split(SUITES["decision-v7"], "train")
    seen = {r["_meta"]["text_sha256"] for r in records if "text_sha256" in r["_meta"]}
    extra = [r for r in v7_train if r["_meta"].get("text_sha256") not in seen]
    records += extra; report["decision_v7_train_added"] = len(extra)

    # 3) synthetic policy cases (trainable families only) and random rule structures (held-out/locked excluded)
    reserved_all = reserved_sem | {semantic_hash(r) for r in records}
    pol = legacy(a.legacy_pairs, f"{a.seed}-control", families=tuple(LEGACY_FAMILIES), excluded=reserved_all)
    records += pol; report["legacy_policy"] = len(pol)
    reserved_all |= {semantic_hash(r) for r in pol}
    trees = {f"rand:{canonical(t)}": t for t in sample_trees(a.structures, f"{a.seed}-structures")}
    comp = compose(a.groups_per_structure, f"{a.seed}-composition", styles=(0, 1, 3, 4), trees=trees)
    comp_kept = [r for r in comp if semantic_hash(r) not in reserved_all]
    records += comp_kept; report["compositional_generated"] = len(comp); report["compositional_kept"] = len(comp_kept)

    # 4) kev's released delta data (dated policy cases, evidence-free cases with uniform targets)
    n2 = read_jsonl(NIGHT2)
    records += n2; report["night2"] = len(n2)

    random.Random(a.seed).shuffle(records)
    with (out / "train.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    src = collections.Counter(r["_meta"]["source"] for r in records)
    manifest = {"records": len(records), "sources": dict(src), "report": dict(report), "seed": a.seed,
                "public_targets": PUBLIC, "legacy_families": LEGACY_FAMILIES, "structures": a.structures,
                "reserved_state_hashes": len(reserved_text), "reserved_semantic_hashes": len(reserved_sem),
                "dataset_revisions": v7["dataset_revisions"], "eval_only_sources_excluded": list(EVAL_ONLY)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
