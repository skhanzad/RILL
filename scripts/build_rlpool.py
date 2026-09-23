"""Novel-task post-training pool: public tasks that appear in NO training pool (A, B or C) and in NO evaluation suite:
SuperGLUE CB / COPA / WiC, GLUE RTE, SciTail, HellaSwag, CLINC150, WinoGrande. Used to post-train (RL, supervised, or
temperature scaling) under task shift. Requests are split into a post-training part and a held-out part.

    python scripts/build_rlpool.py --out data/rlpool --per_source 400
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decisionflow.data import SUITES  # noqa: E402
from kev.data import Source, _dataset, _desc, _instr, _sample, _wrap_state, build  # noqa: E402
from kev.suite import load_split, semantic_hash  # noqa: E402

CB = {"entailment": "The hypothesis follows from the premise", "contradiction": "The hypothesis contradicts the premise",
      "neutral": "The premise neither implies nor contradicts the hypothesis"}


def _cb(split, n, rng):
    ds = _dataset("aps/super_glue:cb", split=split, src=rng)
    keys = list(CB)
    return [{"state": _wrap_state(ex["premise"], rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{ex["hypothesis"]}" How does it relate to the text?', rng),
             "criteria": {k: _desc(v, rng) for k, v in CB.items()}, "label": keys[ex["label"]], "src": "cb"}}} for ex in _sample(ds, n, rng)]


def _copa(split, n, rng):
    ds = _dataset("aps/super_glue:copa", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = "What was the cause?" if ex["question"] == "cause" else "What happened as a result?"
        keys = ["first", "second"]
        out.append({"state": _wrap_state(ex["premise"], rng), "questions": {"alternative": {"type": "choice", "instructions": _instr(q, rng),
                    "criteria": {"first": ex["choice1"], "second": ex["choice2"]}, "label": keys[ex["label"]], "src": "copa"}}})
    return out


def _wic(split, n, rng):
    ds = _dataset("aps/super_glue:wic", split=split, src=rng)
    return [{"state": {"sentence 1": ex["sentence1"], "sentence 2": ex["sentence2"]},
             "questions": {"same_sense": {"type": "noul", "instructions": f'Is the word "{ex["word"]}" used with the same meaning in both sentences?',
                                          "label": ex["label"] == 1, "src": "wic"}}} for ex in _sample(ds, n, rng)]


def _rte(split, n, rng):
    ds = _dataset("nyu-mll/glue:rte", split=split, src=rng)
    return [{"state": _wrap_state(ex["sentence1"], rng), "questions": {"implies": {"type": "noul", "instructions": _instr(f'Does the text imply that "{ex["sentence2"]}"?', rng),
             "label": ex["label"] == 0, "src": "rte"}}} for ex in _sample(ds, n, rng)]


def _scitail(split, n, rng):
    ds = _dataset("allenai/scitail:snli_format", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        q = {"type": "noul", "instructions": _instr(f'Does the text support the statement: "{ex["sentence2"]}"?', rng), "label": ex["gold_label"] == "entailment", "src": "scitail"}
        if rng.random() < 0.5:
            q["criteria"] = {"true": "The text supports the statement", "false": "The text does not support it"}
        out.append({"state": _wrap_state(ex["sentence1"], rng), "questions": {"supports": q}})
    return out


def _hellaswag(split, n, rng):
    ds = _dataset("Rowan/hellaswag", split=split, src=rng)
    out = []
    for ex in _sample(ds, n, rng):
        keys = ["a", "b", "c", "d"]
        out.append({"state": {"activity": ex["activity_label"], "text": ex["ctx"]}, "questions": {"ending": {"type": "choice", "instructions": "Which ending continues the text most plausibly?",
                    "criteria": dict(zip(keys, ex["endings"])), "label": keys[int(ex["label"])], "src": "hellaswag"}}})
    return out


def _clinc(split, n, rng):
    ds = _dataset("clinc/clinc_oos:plus", split=split, src=rng)
    names = ds.features["intent"].names
    out = []
    for ex in _sample(ds, n, rng):
        gold = names[ex["intent"]]
        others = rng.sample([x for x in names if x != gold], 7)
        opts = others + [gold]; rng.shuffle(opts)
        crit = {k: _desc(k.replace("_", " "), rng, p_null=0.5, p_struct=0.0) for k in opts}
        out.append({"state": _wrap_state(ex["text"], rng), "questions": {"intent": {"type": "choice", "instructions": _instr("What does the user want?", rng),
                    "criteria": crit, "label": gold, "src": "clinc150"}}})
    return out


def _winogrande(split, n, rng):
    ds = _dataset("allenai/winogrande:winogrande_xl", split=split, src=rng)
    return [{"state": ex["sentence"], "questions": {"blank": {"type": "choice", "instructions": "Which option correctly fills the blank (_)?",
             "criteria": {"option_1": ex["option1"], "option_2": ex["option2"]}, "label": "option_1" if ex["answer"] == "1" else "option_2", "src": "winogrande"}}}
            for ex in _sample(ds, n, rng) if ex["answer"] in ("1", "2")]


SOURCES = {"cb": _cb, "copa": _copa, "wic": _wic, "rte": _rte, "scitail": _scitail, "hellaswag": _hellaswag, "clinc150": _clinc, "winogrande": _winogrande}
REPOS = {"cb": "aps/super_glue", "copa": "aps/super_glue", "wic": "aps/super_glue", "rte": "nyu-mll/glue", "scitail": "allenai/scitail",
         "hellaswag": "Rowan/hellaswag", "clinc150": "clinc/clinc_oos", "winogrande": "allenai/winogrande"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_source", type=int, default=400)
    ap.add_argument("--heldout_frac", type=float, default=0.25)
    ap.add_argument("--seed", default="rlpool-20260923")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reserved = set()
    for name, d in SUITES.items():
        for split in ("calibration", "development", "test"):
            try:
                reserved |= {r["_meta"]["text_sha256"] for r in load_split(d, split, allow_test=True) if "text_sha256" in r["_meta"]}
            except Exception:
                pass
    train, held, counts = [], [], collections.Counter()
    for name, fn in SOURCES.items():
        recs = build(a.per_source, "train", seed=a.seed, only=(name,), sources={name: (fn, "train", "test")}, repos={name: REPOS[name]})
        recs = [r for r in recs if r["_meta"]["text_sha256"] not in reserved]
        for r in recs:
            r["_meta"].update(variant="clean", group_id=r["_meta"]["id"])
        k = int(round(len(recs) * a.heldout_frac))
        held += recs[:k]; train += recs[k:]
        counts[name] = len(recs)
        print(name, len(recs), flush=True)
    rng = random.Random(a.seed); rng.shuffle(train); rng.shuffle(held)
    for fname, recs in (("train.jsonl", train), ("heldout.jsonl", held)):
        with (out / fname).open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    (out / "manifest.json").write_text(json.dumps({"train": len(train), "heldout": len(held), "per_source": dict(counts), "sources": list(SOURCES)}, indent=1))
    print(len(train), "train /", len(held), "held-out")


if __name__ == "__main__":
    main()
