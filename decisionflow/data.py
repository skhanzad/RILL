"""Training data: frozen kev suites (and larger pools built with the same converters), augmented exactly as kev does
(fresh option permutation / none-of-the-above / distractor per epoch, plus none-present/none-absent minimal pairs)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEV = ROOT / "third_party" / "kev"
if str(KEV) not in sys.path:
    sys.path.insert(0, str(KEV))

from kev.data import augment, materialize, none_pair, source_seed  # noqa: E402
from kev.suite import load_split, read_jsonl  # noqa: E402

from .layout import Encoded, Layout  # noqa: E402

EVALS = KEV / "evals"
SUITES = {
    "decision-v7": EVALS / "v7" / "decision-v7",
    "transfer-v4": EVALS / "v4" / "transfer-v4",
    "transfer-v9": EVALS / "v9" / "transfer-v9",
    "semif-v1": EVALS / "external" / "semif-v1",
    "wanli-v1": EVALS / "external" / "wanli-v1",
    "scienthoon-v1": EVALS / "external" / "scienthoon-v1",
}
NIGHT2 = EVALS / "night2" / "dates_unknowable.jsonl"


def training_requests(spec: str) -> list[dict]:
    """spec: comma-separated items. 'decision-v7' = the suite's training partition; 'night2' = kev's released delta data
    (dated policy cases + evidence-free cases with uniform targets); any other item is a JSONL path of labelled requests."""
    reqs = []
    for item in [s for s in spec.split(",") if s]:
        if item in SUITES:
            reqs += load_split(SUITES[item], "train")
        elif item == "night2":
            reqs += read_jsonl(NIGHT2)
        else:
            reqs += read_jsonl(item)
    for r in reqs:
        r.setdefault("_meta", {}).setdefault("id", json.dumps(r["state"], sort_keys=True)[:64])
    return reqs


def epoch_variants(reqs: list[dict], epoch: int, seed: int, p_none_pair: float = 0.25, p_none: float = 0.1,
                   p_none_distract: float = 0.12, p_distract: float = 0.15) -> list[dict]:
    """kev.train.encode_batch's augmentation policy, request by request, with the same per-item RNG seeding."""
    out = []
    for req in reqs:
        rng = random.Random(source_seed(seed, f"{epoch}:{req['_meta']['id']}"))
        variants = [augment(req, rng, p_none=p_none, p_none_distract=p_none_distract, p_distract=p_distract)]
        if p_none_pair > 0 and rng.random() < p_none_pair:
            variants += none_pair(req, rng)
        for v in variants:
            out.append(materialize(v))
    return out


def encode_all(layout: Layout, recs: list[dict]) -> list[Encoded]:
    return [layout.encode(r) for r in recs]


def token_batches(lengths: list[int], max_tokens: int, max_batch: int, rng: random.Random, shuffle: bool = True):
    """Length-bucketed batches whose padded size (batch x longest) stays within max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, longest = [], [], 0
    for i in order:
        L = lengths[i]
        if cur and (max(longest, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(cur); cur, longest = [], 0
        cur.append(i); longest = max(longest, L)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


# --- parallel epoch preparation (augment -> materialize -> encode) ----------------------------------------------------
_WORKER = {}


def _init_worker(layout, kw):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER["layout"], _WORKER["kw"] = layout, kw


def _prepare_chunk(args):
    reqs, epoch, seed = args
    recs = epoch_variants(reqs, epoch, seed, **_WORKER["kw"])
    return [_WORKER["layout"].encode(r) for r in recs]


def encode_epoch(reqs, epoch, seed, layout, workers=16, chunk=2000, **kw):
    """Same output as encode_all(layout, epoch_variants(reqs, epoch, seed, **kw)) (per-item seeding makes the result
    independent of chunking), computed in parallel worker processes."""
    import multiprocessing as mp
    if workers <= 1 or len(reqs) < 2 * chunk:
        return encode_all(layout, epoch_variants(reqs, epoch, seed, **kw))
    parts = [(reqs[i:i + chunk], epoch, seed) for i in range(0, len(reqs), chunk)]
    with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(layout, kw)) as pool:
        out = []
        for encs in pool.imap(_prepare_chunk, parts):
            out += encs
    return out
