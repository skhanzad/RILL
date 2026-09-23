"""Experiment C training pool = Experiment B pool + additional *non-evaluation* public sources, converted to typed decision
requests in Kev's style: NLI (SNLI, ANLI r1-r3), paraphrase (QQP, MRPC), science/commonsense multiple choice (ARC-Easy,
ARC-Challenge, OpenBookQA, CommonsenseQA; listed as trainable in Kev's own data policy) and reading comprehension
(RACE). Evaluation-only sources (MMLU, Emotion, TweetEval, QNLI/SQuAD, PAWS, SciQ, WANLI) and anything derived from them
are not used; emotion and toxicity datasets are excluded altogether. Every request whose normalised state hash matches
any evaluation/calibration partition is dropped.

    python scripts/build_expC.py --base data/expB/train.jsonl --out data/expC
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402  (kev import path)
from kev.data import MNLI, Source, _arc, _csqa, _dataset, _desc, _instr, _mcq, _openbookqa, _sample, _wrap_state, build, source_seed  # noqa: E402
from kev.suite import load_split, read_jsonl, semantic_hash  # noqa: E402

SUPPORT = {"supported": "The evidence supports the claim", "insufficient": "The evidence neither supports nor contradicts the claim",
           "contradicted": "The evidence contradicts the claim"}


def _nli_formats(premise, hypothesis, label, rng, src):
    """label: 0 entailment, 1 neutral, 2 contradiction. Three renderings, drawn at random."""
    r = rng.random()
    if r < 0.4:
        keys = list(MNLI)
        return {"state": _wrap_state(premise, rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{hypothesis}" How does it relate to the premise?', rng),
                "criteria": {k: _desc(v, rng) for k, v in MNLI.items()}, "label": keys[label], "src": src}}}
    if r < 0.75:
        keys = list(SUPPORT)
        return {"state": _wrap_state(premise, rng), "questions": {"verdict": {"type": "choice", "instructions": _instr(f'Claim: "{hypothesis}" What does the evidence say about the claim?', rng),
                "criteria": {k: _desc(v, rng) for k, v in SUPPORT.items()}, "label": keys[label], "src": src}}}
    q = {"type": "noul", "instructions": _instr(f'Does the text imply that "{hypothesis}"?', rng), "label": label == 0, "src": src + "_yn"}
    if rng.random() < 0.5:
        q["criteria"] = {"true": "The text implies it", "false": "The text contradicts it or does not say"}
    return {"state": _wrap_state(premise, rng), "questions": {"implies": q}}


def _snli(split, n, rng):
    ds = _dataset("stanfordnlp/snli", split=split, src=rng)
    return [_nli_formats(ex["premise"], ex["hypothesis"], ex["label"], rng, "snli") for ex in _sample(ds, n, rng) if ex["label"] >= 0]


def _anli(split, n, rng):
    from datasets import concatenate_datasets
    ds = concatenate_datasets([_dataset("facebook/anli", split=r, src=rng) for r in ("train_r1", "train_r2", "train_r3")])
    return [_nli_formats(ex["premise"], ex["hypothesis"], ex["label"], rng, "anli") for ex in _sample(ds, n, rng) if ex["label"] >= 0]


def _qqp(split, n, rng):
    ds = _dataset("nyu-mll/glue:qqp", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = {"type": "noul", "instructions": _instr(f'Does this question ask the same thing as: "{ex["question2"]}"', rng), "label": ex["label"] == 1, "src": "qqp"}
        if rng.random() < 0.5:
            q["criteria"] = {"true": "Same question, possibly reworded", "false": "A different question"}
        out.append({"state": _wrap_state(ex["question1"], rng), "questions": {"duplicate": q}})
    return out


def _mrpc(split, n, rng):
    ds = _dataset("nyu-mll/glue:mrpc", split=split, src=rng)
    return [{"state": _wrap_state(ex["sentence1"], rng), "questions": {"paraphrase": {"type": "noul", "instructions": _instr(f'Does this sentence say the same thing as: "{ex["sentence2"]}"', rng),
             "label": ex["label"] == 1, "src": "mrpc"}}} for ex in _sample(ds, n, rng)]


def _arc_easy(split, n, rng):
    ds = _dataset("allenai/ai2_arc:ARC-Easy", split=split, src=rng)
    return [_mcq(ex["question"], list(ex["choices"]["label"]), list(ex["choices"]["text"]), ex["answerKey"], "arc_easy", rng) for ex in _sample(ds, n, rng)]


def _race(split, n, rng):
    ds = _dataset("ehovy/race:all", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        article = " ".join(ex["article"].split()[:300])
        labels = ["A", "B", "C", "D"][:len(ex["options"])]
        rec = _mcq(ex["question"], labels, list(ex["options"]), ex["answer"], "race", rng, state_extra={"passage": article})
        rec["questions"]["answer"]["instructions"] = "Which option answers the question about the passage?"
        out.append(rec)
    return out


EXTRA = {"snli": (_snli, 60000), "anli": (_anli, 60000), "qqp": (_qqp, 40000), "mrpc": (_mrpc, 4000), "arc": (_arc, 2000),
         "arc_easy": (_arc_easy, 3000), "openbookqa": (_openbookqa, 5000), "csqa": (_csqa, 10000), "race": (_race, 30000)}
REPOS = {"snli": "stanfordnlp/snli", "anli": "facebook/anli", "qqp": "nyu-mll/glue", "mrpc": "nyu-mll/glue", "arc": "allenai/ai2_arc",
         "arc_easy": "allenai/ai2_arc", "openbookqa": "allenai/openbookqa", "csqa": "tau/commonsense_qa", "race": "ehovy/race"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/expB/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="expC-20260923")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
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
    records = read_jsonl(a.base)
    report = collections.Counter({"base_records": len(records)})
    for name, (fn, n) in EXTRA.items():
        recs = build(n, "train", seed=a.seed, only=(name,), sources={name: (fn, "train", "test")}, repos={name: REPOS[name]})
        kept = [r for r in recs if r["_meta"]["text_sha256"] not in reserved_text and semantic_hash(r) not in reserved_sem]
        for r in kept:
            r["_meta"]["variant"] = "clean"; r["_meta"].setdefault("group_id", r["_meta"]["id"])
        records += kept
        report[f"{name}_built"] = len(recs); report[f"{name}_dropped_reserved"] = len(recs) - len(kept)
        print(f"{name}: built {len(recs)} kept {len(kept)}", flush=True)
    random.Random(a.seed).shuffle(records)
    with (out / "train.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    src = collections.Counter(r["_meta"]["source"] for r in records)
    manifest = {"records": len(records), "sources": dict(src), "report": dict(report), "seed": a.seed, "extra_sources": list(EXTRA),
                "excluded": ["mmlu", "emotion", "tweet_eval", "qnli", "squad", "paws", "sciq", "wanli", "emotion/toxicity datasets"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
