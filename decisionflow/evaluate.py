"""Batched evaluation of DecisionFlow on frozen suites, scored with kev's own metric code (kev.benchmark / kev.metrics),
so every system in the paper is measured by exactly the same functions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import SUITES  # noqa: F401  (sets up the kev import path)
from kev.api import SystemOneRequest, question_keys, to_record  # noqa: E402
from kev.benchmark import prediction_rows, summarize  # noqa: E402
from kev.data import api_request  # noqa: E402
from kev.suite import load_split, read_manifest  # noqa: E402

from .layout import Layout, collate  # noqa: E402


# Inference option: present the options of choice questions in a canonical order (sorted by option name). A choice
# question's options are a set, so this makes every decision exactly invariant to the order the caller lists them in.
CANONICAL_ORDER = False


def request_to_rec(record: dict, canonical: bool | None = None):
    """Label-free serving path: the API request body only (kev.data.api_request), rendered by kev.api.to_record."""
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(record)))
    canonical = CANONICAL_ORDER if canonical is None else canonical
    for q, m in zip(rec["questions"], meta):
        q["qtype"] = m["type"]
        if canonical and m["type"] == "choice":
            order = sorted(range(len(m["keys"])), key=lambda i: m["keys"][i])
            q["options"] = [q["options"][i] for i in order]
            m["keys"] = [m["keys"][i] for i in order]
    return rec, meta


@torch.no_grad()
def predict_records(model, layout: Layout, records: list[dict], device: str, nfe: int = 1, max_tokens: int = 24000,
                    max_batch: int = 64, temperature: float = 1.0, noise: str = "zero", seed: int = 0):
    """Returns a list of kev-style predictions ({"probabilities": {qid: {key: p}}, "logits": ..., ...}) aligned with
    `records`, plus the number of records whose state had to be truncated."""
    model.eval()
    encs, metas = [], []
    for r in records:
        rec, meta = request_to_rec(r)
        encs.append(layout.encode(rec)); metas.append(meta)
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    out = [None] * len(encs)
    gen = torch.Generator(device=device).manual_seed(seed)
    i = 0
    while i < len(order):
        j, longest = i, 0
        while j < len(order) and j - i < max_batch and max(longest, len(encs[order[j]].ids)) * (j - i + 1) <= max_tokens:
            longest = max(longest, len(encs[order[j]].ids)); j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        batch = collate([encs[k] for k in idx], layout.pad).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            p, z = model.readout(batch, nfe=nfe, noise=noise, generator=gen, return_logits=True)
        z = z.float() / temperature
        p = model.probs(z, batch) if temperature != 1.0 else p.float()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000 / len(idx)
        p, z = p.cpu().numpy(), z.cpu().numpy()
        starts, lens = batch.q_start.cpu().numpy(), batch.q_len.cpu().numpy()
        qn = 0
        for k in idx:
            probs, logits = {}, {}
            for m in metas[k]:
                s, n = starts[qn], lens[qn]
                pk = p[s:s + n].astype(np.float64); pk = pk / pk.sum()
                probs[m["id"]] = dict(zip(m["keys"], pk.tolist()))
                logits[m["id"]] = dict(zip(m["keys"], z[s:s + n].astype(np.float64).tolist()))
                qn += 1
            out[k] = {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                      "latency_ms": dt, "input_tokens": len(encs[k].ids)}
        i = j
    return out


def score_suite(model, layout, suite: str, split: str, device: str, out_dir: Path | None = None, **kw):
    """Predict a suite partition and summarise it with kev.benchmark.summarize. Returns (report, rows)."""
    from .data import SUITES
    d = SUITES[suite]
    records = load_split(d, split, allow_test=(split == "test"))
    manifest = read_manifest(d)
    preds = predict_records(model, layout, records, device, **kw)
    rows = []
    for r, pr in zip(records, preds):
        rows += prediction_rows(r, pr)
    report = summarize(rows, 1.0, tuple(manifest.get("holdout_sources", [])))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{suite}.{split}.rows.json").write_text(json.dumps(rows))
        (out_dir / f"{suite}.{split}.report.json").write_text(json.dumps(report, indent=1))
    return report, rows


def brief(report: dict) -> dict:
    c = report["clean"]
    keys = ["n", "acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc"]
    return {k: (round(c[k], 4) if isinstance(c.get(k), float) else c.get(k)) for k in keys}
