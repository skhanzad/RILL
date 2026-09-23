"""Stage 1: conditional flow matching on the answer track.

    python -m decisionflow.train --backbone jhu-clsp/ettin-encoder-68m --train decision-v7,night2 --out runs/df-68m-A

For every question with target distribution y over its K options, the answer-track target is x_1 = y (yes/no, choice)
or cumsum(y) (score). With x_0 ~ N(0, I) and t ~ pi(t), x_t = (1 - t) x_0 + t x_1, and the network predicts
x1_hat = enc(softmax(z)). The loss is  || x1_hat - x_1 ||^2  (flow matching with v = (x1_hat - x_t)/(1-t), reweighted by
(1-t)^2)  +  lambda_ce * CE(y, softmax(z)).  With probability p_sc the network is first run without a self-condition and
then conditioned on its own detached estimate (masked self-conditioning channel).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import encode_all, encode_epoch, epoch_variants, token_batches, training_requests
from .evaluate import brief, score_suite
from .layout import Layout, collate
from .model import DecisionFlow, seg_log_softmax, seg_sum


def flow_matching_loss(model, batch, cache, t, p_sc, lam_ce, lam_flow, gen_rng):
    y = batch.target                                   # (S,) target distribution, concatenated per question
    NQ = len(batch.q_len)
    x1 = model.encode_target(y, batch)
    eps = torch.randn_like(x1)
    tt = t[batch.slot_b]
    xt = (1 - tt) * eps + tt * x1
    c, flag = torch.zeros_like(xt), 0.0
    if gen_rng.random() < p_sc:
        with torch.no_grad():
            z0 = model.flow_branch(cache, batch, xt, c, t, sc_flag=0.0)
            c = model.encode_target(model.probs(z0.float(), batch), batch).detach(); flag = 1.0
    z = model.flow_branch(cache, batch, xt, c, t, sc_flag=flag).float()
    logp = seg_log_softmax(z, batch.slot_q, NQ)
    x1_hat = model.encode_target(logp.exp(), batch)
    brier_q = seg_sum((x1_hat - x1) ** 2, batch.slot_q, NQ)
    ce_q = -seg_sum(y * logp, batch.slot_q, NQ)
    loss_q = lam_flow * brier_q + lam_ce * ce_q
    B = batch.ids.shape[0]
    nq = torch.zeros(B, device=z.device).index_add(0, batch.q_b, torch.ones_like(loss_q))
    per_rec = torch.zeros(B, device=z.device).index_add(0, batch.q_b, loss_q) / nq.clamp_min(1)
    return per_rec.mean(), {"brier": brier_q.mean().item(), "ce": ce_q.mean().item()}


def build_model(args, device):
    model = DecisionFlow(args.backbone, dit_layers=args.dit_layers, convnext_blocks=args.convnext,
                         score_encoding=args.score_encoding, attn_dropout=args.attn_dropout)
    return model.to(device)


def model_config(args):
    return {"backbone": args.backbone, "dit_layers": args.dit_layers, "convnext_blocks": args.convnext,
            "score_encoding": args.score_encoding}


def save(model, cfg, path):
    torch.save({"state_dict": model.state_dict(), "config": {**cfg, "dit_layers": model.n_dit, "calibrator": model.calibrator is not None}}, path)


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = DecisionFlow(cfg["backbone"], dit_layers=cfg["dit_layers"], convnext_blocks=cfg["convnext_blocks"],
                         score_encoding=cfg["score_encoding"])
    if cfg.get("calibrator"):
        model.add_calibrator()
    model.load_state_dict(ck["state_dict"])
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    return model.to(device).eval(), tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--train", default="decision-v7,night2")
    ap.add_argument("--init", default="", help="warm-start from a DecisionFlow checkpoint (.pt)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_new", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--max_state", type=int, default=512)
    ap.add_argument("--p_t0", type=float, default=0.5, help="probability of training exactly at t=0 (the System-1 readout)")
    ap.add_argument("--p_sc", type=float, default=0.5, help="self-conditioning probability")
    ap.add_argument("--lam_ce", type=float, default=1.0)
    ap.add_argument("--lam_flow", type=float, default=1.0)
    ap.add_argument("--dit_layers", type=int, default=None)
    ap.add_argument("--convnext", type=int, default=2)
    ap.add_argument("--score_encoding", default="cdf", choices=["cdf", "onehot"])
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--p_none_pair", type=float, default=0.25)
    ap.add_argument("--eval_suites", default="decision-v7,transfer-v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: use only this many training requests")
    ap.add_argument("--workers", type=int, default=16, help="processes for per-epoch augmentation + encoding")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(args.backbone)
    layout = Layout(tok, max_state=args.max_state)
    eval_layout = Layout(tok, max_state=7000)
    model = build_model(args, dev)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False); model.load_state_dict(ck["state_dict"])
    else:
        model.init_special_tokens(tok)
    cfg = model_config(args)
    nparams = sum(p.numel() for p in model.parameters())
    reqs = training_requests(args.train)
    if args.limit:
        reqs = random.Random(args.seed).sample(reqs, min(args.limit, len(reqs)))
    print(f"model {args.backbone}: {nparams/1e6:.1f}M params, n_txt={model.n_txt} n_dit={model.n_dit}; {len(reqs)} training requests", flush=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "params": nparams, "n_txt": model.n_txt, "n_dit": model.n_dit}, indent=1))

    # estimate steps for the schedule from the first epoch's variants
    first = encode_epoch(reqs, 0, args.seed, layout, workers=args.workers, p_none_pair=args.p_none_pair)
    steps_per_epoch = len(token_batches([len(e.ids) for e in first], args.max_tokens, args.max_batch, rng))
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr_new, args.wd), betas=(0.9, 0.98), eps=1e-6)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm))))))
    log = open(out / "train.log", "a")
    step, t_start, best = 0, time.time(), None
    history = []
    for epoch in range(args.epochs):
        encs = first if epoch == 0 else encode_epoch(reqs, epoch, args.seed, layout, workers=args.workers, p_none_pair=args.p_none_pair)
        batches = token_batches([len(e.ids) for e in encs], args.max_tokens, args.max_batch, rng)
        model.train()
        run = {"loss": 0.0, "brier": 0.0, "ce": 0.0, "n": 0}
        for bi, idx in enumerate(batches):
            batch = collate([encs[i] for i in idx], tok.pad_token_id).to(dev)
            B = batch.ids.shape[0]
            t = torch.where(torch.rand(B, device=dev) < args.p_t0, torch.zeros(B, device=dev), torch.rand(B, device=dev) * 0.999)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache = model.text_branch(batch)
                loss, parts = flow_matching_loss(model, batch, cache, t, args.p_sc, args.lam_ce, args.lam_flow, rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run["loss"] += loss.item(); run["brier"] += parts["brier"]; run["ce"] += parts["ce"]; run["n"] += 1
            if step % 50 == 0:
                msg = (f"ep {epoch} step {step}/{total} loss {run['loss']/run['n']:.4f} brier {run['brier']/run['n']:.4f} "
                       f"ce {run['ce']/run['n']:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t_start:.0f}s")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                run = {"loss": 0.0, "brier": 0.0, "ce": 0.0, "n": 0}
        rec = {"epoch": epoch, "step": step, "time_s": time.time() - t_start}
        for suite in [s for s in args.eval_suites.split(",") if s]:
            report, _ = score_suite(model, eval_layout, suite, "development", dev)
            rec[suite] = {**brief(report), "objective": report["objective"]}
        history.append(rec)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        save(model, cfg, out / "last.pt")
        sel = sum(rec[s]["objective"] for s in args.eval_suites.split(",") if s)
        if best is None or sel > best:
            best = sel; save(model, cfg, out / "best.pt")
            (out / "best.json").write_text(json.dumps(rec, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
