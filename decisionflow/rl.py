"""Stage 2: reinforcement post-training for calibrated decisions (outcome-reward policy optimisation).

Policy. For a request, the flow sampler is a stochastic policy over the answer track: x_0 ~ N(0, I), then for
n = 0..N-2 the Euler-Maruyama transition  x_{n+1} ~ N( x_n + dt (x1_hat_n - x_n) / (1 - t_n),  sigma^2 dt I ),  with
x1_hat_n the network's posterior-mean estimate (self-conditioned on x1_hat_{n-1}). The executed decision for question q
is a_q = argmax p^{(N-1)}_q, and the system states the System-1 confidence  c_q = pbar_q(a_q),  where pbar is the
deterministic t=0 readout that the API returns.

Reward (outcome only; the label is never shown to the policy):   R = z - (c - z)^2,   z = y(a)  (= 1[a correct]).
For any fixed decision, E[R | a] is maximised uniquely at c = P(a correct | request) (Brier properness); equivalently,
R/2 + const is the expected utility of executing a when an externally drawn threshold tau ~ U(0,1) is below c
(execute: +1-tau if correct, -tau if wrong; abstain: 0).

Update.  L = -E[ A_i * mean_n log pi(x_{n+1} | x_n) ]          (group-relative advantages over G rollouts, GRPO)
           + lam_cal * E[ (pbar(a_i) - z_i)^2 ]              (pathwise: the reward's confidence term, proper under bandit feedback)
           + beta * KL( pbar_ref || pbar )                    (anchor to the Stage-1 readout)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import SUITES, encode_all, epoch_variants, token_batches, training_requests
from .evaluate import brief, score_suite
from .layout import Layout, collate
from .model import seg_log_softmax, seg_softmax, seg_sum
from .train import load, save
from kev.suite import load_split  # noqa: E402


def repeat_cache(cache, G):
    return {"h": cache["h"].repeat(G, 1, 1),
            "masks": {k: v.repeat(G, 1, 1, 1) for k, v in cache["masks"].items()},
            "rope": {k: (c.repeat(G, 1, 1), s.repeat(G, 1, 1)) for k, (c, s) in cache["rope"].items()}}


def question_argmax(p, batch):
    """Index (within the question) of the most probable option for each question."""
    NQ = len(batch.q_len)
    pmax = torch.full((NQ,), -1.0, device=p.device).scatter_reduce(0, batch.slot_q, p, "amax", include_self=True)
    is_max = p >= pmax[batch.slot_q]
    local = torch.arange(p.numel(), device=p.device) - batch.q_start[batch.slot_q]
    big = torch.where(is_max, local, torch.full_like(local, 10 ** 6))
    return torch.full((NQ,), 10 ** 6, device=p.device, dtype=torch.long).scatter_reduce(0, batch.slot_q, big, "amin", include_self=True)


@torch.no_grad()
def rollout(model, batch, cache, nfe, sigma, gen):
    """Returns the trajectory (inputs to each step), the final probabilities and the per-question decisions."""
    S = batch.slot_b.numel(); B = batch.ids.shape[0]; dev = batch.ids.device
    ts = torch.linspace(0, 1, nfe + 1, device=dev)
    x = torch.randn(S, device=dev, generator=gen)
    c, flag = torch.zeros(S, device=dev), 0.0
    xs, cs, flags = [], [], []
    p = None
    for n in range(nfe):
        xs.append(x); cs.append(c); flags.append(flag)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.flow_branch(cache, batch, x, c, ts[n].expand(B), sc_flag=flag).float()
        p = model.probs(z, batch)
        x1 = model.encode_target(p, batch)
        if n < nfe - 1:
            dt = ts[n + 1] - ts[n]
            mu = x + dt * (x1 - x) / (1 - ts[n])
            x = mu + sigma * dt.sqrt() * torch.randn(S, device=dev, generator=gen)
        c, flag = x1, 1.0
    xs.append(x)
    return {"xs": xs, "cs": cs, "flags": flags, "p_final": p, "a": question_argmax(p, batch)}


def trajectory_logprob(model, batch, cache, traj, nfe, sigma):
    """Per-question mean log-density of the stochastic transitions x_n -> x_{n+1}, n = 0..N-2 (teacher-forced)."""
    B = batch.ids.shape[0]; dev = batch.ids.device; NQ = len(batch.q_len)
    ts = torch.linspace(0, 1, nfe + 1, device=dev)
    lp = torch.zeros(NQ, device=dev)
    for n in range(nfe - 1):
        x, c, flag = traj["xs"][n], traj["cs"][n], traj["flags"][n]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.flow_branch(cache, batch, x, c, ts[n].expand(B), sc_flag=flag).float()
        x1 = model.encode_target(model.probs(z, batch), batch)
        dt = ts[n + 1] - ts[n]
        mu = x + dt * (x1 - x) / (1 - ts[n])
        var = (sigma ** 2) * dt
        ll = -0.5 * (traj["xs"][n + 1] - mu) ** 2 / var          # constants cancel in the advantage-weighted sum
        lp = lp + seg_sum(ll, batch.slot_q, NQ) / batch.q_len.float()
    return lp / max(1, nfe - 1)


def readout(model, batch, cache):
    S = batch.slot_b.numel(); B = batch.ids.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.flow_branch(cache, batch, torch.zeros(S, device=batch.ids.device), torch.zeros(S, device=batch.ids.device),
                              torch.zeros(B, device=batch.ids.device), sc_flag=0.0).float()
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="Stage-1 checkpoint (.pt)")
    ap.add_argument("--pool", default="calibration", help="'calibration' (decision-v7 calibration partition) or training spec / JSONL")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--G", type=int, default=8, help="rollouts per request (group size)")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--lam_pg", type=float, default=1.0)
    ap.add_argument("--lam_cal", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=2500, help="token budget per batch before the G-fold expansion")
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--augment", type=int, default=1, help="kev augmentation of the pool each epoch")
    ap.add_argument("--eval_suites", default="decision-v7,transfer-v4")
    ap.add_argument("--calibrator", type=int, default=0, help="1: freeze the decision network and train only an input-dependent temperature head")
    ap.add_argument("--lr_calib", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    model, tok, cfg = load(a.init, "cuda")
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    layout = Layout(tok, max_state=512)
    eval_layout = Layout(tok, max_state=7000)
    pool = load_split(SUITES["decision-v7"], "calibration") if a.pool == "calibration" else training_requests(a.pool)
    print(f"RL pool: {len(pool)} requests", flush=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    if a.calibrator:
        for p_ in model.parameters():
            p_.requires_grad_(False)
        calib = model.add_calibrator()
        opt = torch.optim.AdamW(calib.parameters(), lr=a.lr_calib, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0)
        print(f"calibrator mode: {sum(p_.numel() for p_ in calib.parameters())} trainable parameters; decision network frozen", flush=True)
    else:
        opt = torch.optim.AdamW(model.param_groups(a.lr, a.lr_new, 0.0), betas=(0.9, 0.98), eps=1e-6)
    log = open(out / "rl.log", "a")

    def evaluate(tag):
        rec = {"tag": tag}
        for suite in [s for s in a.eval_suites.split(",") if s]:
            report, _ = score_suite(model, eval_layout, suite, "development", "cuda")
            rec[suite] = {**brief(report), "objective": report["objective"]}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        return rec

    history = [evaluate("stage1")]
    t0 = time.time(); step = 0
    for epoch in range(a.epochs):
        recs = epoch_variants(pool, epoch, a.seed + 1000, p_none_pair=0.25) if a.augment else epoch_variants(pool, 0, a.seed, 0.0, 0.0, 0.0, 0.0)
        encs = encode_all(layout, recs)
        batches = token_batches([len(e.ids) for e in encs], a.max_tokens, a.max_batch, rng)
        stats = {"R": 0.0, "acc": 0.0, "conf": 0.0, "pg": 0.0, "cal": 0.0, "kl": 0.0, "n": 0}
        for idx in batches:
            model.train()
            base = collate([encs[i] for i in idx], tok.pad_token_id).to("cuda")
            rep = collate([encs[i] for i in idx] * a.G, tok.pad_token_id).to("cuda")
            NQ, NQr = len(base.q_len), len(rep.q_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cache = model.text_branch(base)
            cache_r = repeat_cache(cache, a.G)
            model.eval()
            traj = rollout(model, rep, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in cache_r.items()}, a.nfe, a.sigma, gen)
            model.train()
            # outcome of each sampled decision: z = y(a)  (1[a = label] for hard labels)
            y = rep.target
            a_idx = traj["a"]
            z = y[rep.q_start + a_idx]
            # stated confidence = deterministic System-1 readout of the chosen option (with grad for the pathwise term)
            zb = readout(model, base, cache)
            pbar = seg_softmax(zb, base.slot_q, NQ)
            pbar_r = pbar.repeat(a.G)
            conf = pbar_r[rep.q_start + a_idx]
            R = (z - (conf.detach() - z) ** 2)
            Rg = R.view(a.G, NQ)
            adv = ((Rg - Rg.mean(0, keepdim=True)) / (Rg.std(0, keepdim=True) + 1e-4)).view(-1)
            lp = trajectory_logprob(model, rep, cache_r, traj, a.nfe, a.sigma)
            loss_pg = -(adv.detach() * lp).mean()
            loss_cal = ((conf - z) ** 2).mean()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zref = ref.flow_branch(ref.text_branch(base), base, torch.zeros_like(zb), torch.zeros_like(zb),
                                       torch.zeros(base.ids.shape[0], device="cuda"), sc_flag=0.0).float()
            lq_ref = seg_log_softmax(zref, base.slot_q, NQ)
            lq = seg_log_softmax(zb, base.slot_q, NQ)
            kl = seg_sum(lq_ref.exp() * (lq_ref - lq), base.slot_q, NQ).mean()
            loss = a.lam_pg * loss_pg + a.lam_cal * loss_cal + a.beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            stats["R"] += R.mean().item(); stats["acc"] += (z > 0.5).float().mean().item(); stats["conf"] += conf.mean().item()
            stats["pg"] += loss_pg.item(); stats["cal"] += loss_cal.item(); stats["kl"] += kl.item(); stats["n"] += 1
            if step % 20 == 0:
                n = stats["n"]
                msg = (f"ep {epoch} step {step} R {stats['R']/n:.4f} acc {stats['acc']/n:.3f} conf {stats['conf']/n:.3f} "
                       f"pg {stats['pg']/n:.4f} cal {stats['cal']/n:.4f} kl {stats['kl']/n:.4f} {time.time()-t0:.0f}s")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                stats = {k: 0.0 for k in stats}; stats["n"] = 0
        history.append(evaluate(f"epoch{epoch}"))
        save(model, cfg, out / f"epoch{epoch}.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
