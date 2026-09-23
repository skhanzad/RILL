"""Build every LaTeX table, pgfplots data file and number macro of the paper from saved results. Nothing in the paper's
tables is typed by hand.

    python scripts/aggregate.py && python scripts/pool_report.py --tag 68B && python scripts/ablation_report.py \
        && python scripts/make_tables.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
KEV_LOCKED = ROOT / "third_party/kev/runs/locked"
MACROS = []


def macro(name, val):
    MACROS.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def fmt(x, d=3, pct=False, lead=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if pct:
        return f"{100 * x:.1f}"
    s = f"{x:.{d}f}"
    return s[1:] if (lead and s.startswith("0.")) else s


def params_str(p):
    if p is None:
        return "--"
    return f"{p / 1e9:.1f}B" if p >= 1e9 else f"{p / 1e6:.0f}M"


def kev9b():
    s = json.loads((KEV_LOCKED / "kev-9b-night2-du-ungated" / "summary.json").read_text())
    out = {}
    for suite, key in (("decision-v7", "decision"), ("transfer-v4", "transfer")):
        c = s["suites"][key]["clean"]
        out[suite] = {"raw": {k: c.get(k) for k in ("n", "acc", "nll", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")},
                      "variants": {}, "perm_flip": s["suites"][key].get("permutation", {}).get("flip_rate")}
    return out


def best_idx(values, higher):
    vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not vals:
        return set()
    b = max(v for _, v in vals) if higher else min(v for _, v in vals)
    return {i for i, v in vals if abs(v - b) < 1e-9}


def rows_table(order, agg, cols, groups, ncols, bold=True):
    """cols: list of (suite, getter, higher_is_better, formatter)."""
    vals = {o: [g(agg[o]["suites"].get(s, {})) for s, g, _, _ in cols] for o in order}
    best = [best_idx([vals[o][j] for o in order], hib) if (bold and hib is not None) else set() for j, (_, _, hib, _) in enumerate(cols)]
    lines, last = [], None
    for i, o in enumerate(order):
        g = groups.get(agg[o]["info"].get("family"), "")
        if g != last:
            if last is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\emph{{{g}}}}}\\\\")
            last = g
        cells = []
        for j, (_, _, _, f) in enumerate(cols):
            c = f(vals[o][j])
            cells.append(f"\\textbf{{{c}}}" if i in best[j] and vals[o][j] is not None else c)
        lines.append(f"{agg[o]['info']['label']} & {params_str(agg[o]['info'].get('params'))} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines) + "\n"


GROUPS = {"rill": "\\method (ours, data C)", "rill-rl": "\\method (ours, data C) + RL calibrator", "rill-canon": "\\method (ours, data C) + RL calibrator", "rill-A": "\\method trained on Kev's data (controlled)",
          "rill-B": "\\method on Kev's sources, 26$\\times$ data", "kev": "Kev: LoRA + pointer head on Qwen3.5, with its shipped temperature",
          "semif": "SemIf: option-letter logits", "layla": "Layla on-device LLMs (letter logits)", "nli": "NLI zero-shot classifiers",
          "layla-gen": "Layla, generative use", "gen": "Generative use"}


def raw(m):
    return lambda e: (e.get("raw") or {}).get(m)


def main():
    agg = json.loads((RES / "aggregate.json").read_text())
    for n, info in json.loads((RES / "registry.json").read_text()).items():     # labels may change after aggregation
        if n in agg:
            agg[n]["info"]["label"] = info["label"]
    # systems whose checkpoints ship a temperature (Kev) are shown as released: every table and macro uses the served
    # probabilities; the unscaled logits stay available as raw_unscaled (appendix temperature table, *Raw* macros)
    for n, a in agg.items():
        for suite, e in a["suites"].items():
            if e.get("served"):
                e["raw_unscaled"], e["raw"] = e["raw"], {**e["raw"], **e["served"]}
                e["unknowable_unscaled"], e["unknowable"] = e.get("unknowable"), e.get("served_unknowable")
    agg["Kev-9B"] = {"info": {"label": "Kev-9B$^\\dagger$", "params": 8.953803264e9 + 43.3e6, "family": "kev"}, "suites": kev9b()}
    have = lambda names: [n for n in names if n in agg and agg[n]["suites"]]

    # ------------------------------------------------------------ Table 1: main
    order = have(["Rill-32M", "Rill-68M", "Rill-150M", "Rill-32M+RL", "Rill-68M+RL", "Rill-150M+RL",
                  "Kev-0.8B", "Kev-4B", "Kev-9B", "SemIf-Qwen3-0.6B", "SemIf-MiniCPM5-2B", "SemIf-Qwen3.5-4B",
                  "Layla-TinyLlama-1.1B", "Layla-Qwen1.5-1.8B", "Layla-Phi2-2.7B", "Layla-Mistral-7B",
                  "NLI-DeBERTa-base", "NLI-BART-large", "NLI-DeBERTa-large"])
    f3, f2, fp = (lambda v: fmt(v, 3)), (lambda v: fmt(v, 2)), (lambda v: fmt(v, pct=True))
    cols = [("decision-v7", raw("acc"), True, f3), ("decision-v7", raw("brier"), False, f3),
            ("transfer-v4", raw("acc"), True, f3), ("transfer-v4", raw("nll"), False, f3), ("transfer-v4", raw("brier"), False, f3),
            ("transfer-v4", raw("ece"), False, f3), ("transfer-v4", raw("confident_error_rate"), None, fp),
            ("transfer-v4", raw("coverage_at_5pct_error"), True, f2)]
    (PAPER / "tables" / "main_rows.tex").write_text(rows_table(order, agg, cols, GROUPS, 10))
    order_d = have(["Rill-68M-A", "Rill-150M-A", "Rill-68M-B", "Rill-68M-B+RL", "Rill-32M", "Rill-68M", "Rill-150M", "Kev-0.8B"])
    (PAPER / "tables" / "data_rows.tex").write_text(rows_table(order_d, agg, cols, GROUPS, 10, bold=False))

    # ------------------------------------------------------------ Table 2: hallucination probes + external suites
    order2 = have(["Rill-32M", "Rill-68M", "Rill-150M", "Rill-32M+RL", "Rill-68M+RL", "Rill-150M+RL", "Rill-68M+RL-canon", "Kev-0.8B", "Kev-4B",
                   "SemIf-Qwen3-0.6B", "SemIf-MiniCPM5-2B", "SemIf-Qwen3.5-4B", "Layla-Mistral-7B",
                   "NLI-DeBERTa-base", "NLI-BART-large", "NLI-DeBERTa-large"])
    unk = lambda e: ((e.get("unknowable") or {}).get("share_at_0_9"))
    nab = lambda e: (e.get("variants") or {}).get("none_absent")
    npr = lambda e: (e.get("variants") or {}).get("none_present")
    flip = lambda e: e.get("perm_flip")
    # probes that a model can game by always (or never) choosing one answer are not bolded
    cols2 = [("transfer-v9", unk, None, fp), ("transfer-v4", nab, None, f3), ("transfer-v4", npr, None, f3), ("transfer-v4", flip, False, fp),
             ("semif-v1", raw("acc"), True, f3), ("wanli-v1", raw("acc"), True, f3), ("scienthoon-v1", raw("acc"), True, f3)]
    # the calibrator changes no decision, so its rows would repeat Stage 1 except for the confidence-based "unknowable"
    # column: show that column as "without / with calibrator" on the Stage-1 rows instead
    order2 = [o for o in order2 if o not in ("Rill-32M+RL", "Rill-68M+RL", "Rill-150M+RL")]
    agg2 = json.loads(json.dumps({o: agg[o] for o in order2}))
    agg2["Rill-68M+RL-canon"]["info"]["family"] = "rill"
    lines2 = rows_table(order2, agg2, cols2, GROUPS, 9).split("\n")
    for base, rlname in (("Rill-32M", "Rill-32M+RL"), ("Rill-68M", "Rill-68M+RL"), ("Rill-150M", "Rill-150M+RL")):
        if base in agg and rlname in agg:
            lab_ = agg[base]["info"]["label"] + " &"
            u0 = unk(agg[base]["suites"].get("transfer-v9", {})); u1 = unk(agg[rlname]["suites"].get("transfer-v9", {}))
            for k, l in enumerate(lines2):
                if l.startswith(lab_):
                    cells = l.split(" & ")
                    cells[2] = f"{fp(u0)} / {fp(u1)}"
                    lines2[k] = " & ".join(cells)
    (PAPER / "tables" / "halluc_rows.tex").write_text("\n".join(lines2))

    # generative use: invalid-output rate (format hallucination)
    gen_rows = []
    for n in ["Layla-TinyLlama-1.1B-gen", "Layla-Qwen1.5-1.8B-gen", "Layla-Phi2-2.7B-gen", "Layla-Mistral-7B-gen", "Qwen3.5-4B-gen"]:
        d = ROOT / json.loads((RES / "registry.json").read_text())[n]["dir"]
        tot = inv = 0; accs = {}
        for suite in ("decision-v7", "transfer-v4"):
            p = d / f"{suite}.test.report.json"
            if p.exists():
                r = json.loads(p.read_text()); g = r.get("generative") or {}
                tot += g.get("total", 0); inv += g.get("invalid", 0); accs[suite] = r["clean"]["acc"]
        if tot:
            gen_rows.append((n, inv / tot, accs.get("decision-v7"), accs.get("transfer-v4")))
    if gen_rows:
        macro("GenInvalidMin", fmt(min(r[1] for r in gen_rows), pct=True))
        macro("GenInvalidMax", fmt(max(r[1] for r in gen_rows), pct=True))
    import collections
    gkeys = {"Layla-TinyLlama-1.1B-gen": "LaylaTiny", "Layla-Qwen1.5-1.8B-gen": "LaylaQwen", "Layla-Phi2-2.7B-gen": "LaylaPhi",
             "Layla-Mistral-7B-gen": "LaylaSeven", "Qwen3.5-4B-gen": "QwenFour"}
    for n, iv, a1, a2 in gen_rows:
        k = gkeys[n]
        macro(f"GenInv{k}", fmt(iv, pct=True))
        d = ROOT / json.loads((RES / "registry.json").read_text())[n]["dir"]
        c = collections.Counter()
        for suite in ("decision-v7", "transfer-v4"):
            g = d / f"{suite}.test.generations.json"
            if g.exists():
                c.update(x["text"].strip()[:1] for x in json.loads(g.read_text()) if x.get("valid"))
        if c:
            letter, cnt = c.most_common(1)[0]
            macro(f"GenTop{k}", fmt(cnt / sum(c.values()), pct=True)); macro(f"GenTopLetter{k}", letter)
    lab = json.loads((RES / "registry.json").read_text())
    (PAPER / "tables" / "gen_rows.tex").write_text("".join(
        f"{lab[n]['label']} & {fmt(iv, pct=True)} & {fmt(a1, 3)} & {fmt(a2, 3)}\\\\\n" for n, iv, a1, a2 in gen_rows))

    # ------------------------------------------------------------ Table 3: efficiency
    lat = {}
    for f in (RES / "latency").glob("*.json"):
        lat[f.stem] = json.loads(f.read_text())
    eff = [("\\method-32M", "cpu4-rill-32m-C", "cpu4-rill-32m-C-g8w8", 34.2e6),
           ("\\method-68M", "cpu4-rill-68m-C", "cpu4-rill-68m-C-g8w8", 72.9e6),
           ("\\method-150M", "cpu4-rill-150m-C", "cpu4-rill-150m-C-g8w8", 158.2e6),
           ("Kev-0.8B", "cpu4-kev-0.8b", None, 752.9e6),
           ("SemIf (Qwen3-0.6B)$^\\ddagger$", "cpu4-semif-qwen3-0.6b", None, 596.0e6),
           ("SemIf (MiniCPM5-2B)", "cpu4-semif-minicpm5-2b", None, 2516.8e6),
           ("SemIf (Qwen3.5-4B)", "cpu4-semif-qwen3.5-4b", None, 4205.8e6)]
    er = []
    for name, cpu, cpu8, pc in eff:
        c = lat.get(cpu); c8 = lat.get(cpu8) if cpu8 else None
        if c is None:
            continue
        q8 = f"{c8['single']['median_ms']:,.0f}" if c8 else "--"
        mb8 = f"{c8['info']['model_mb']:,.0f}" if c8 and c8["info"].get("model_mb") else "--"
        er.append(f"{name} & {params_str(pc)} & {c['single']['median_ms']:,.0f} & {c['three']['median_ms']:,.0f} & {q8} & {mb8}\\\\")
    (PAPER / "tables" / "eff_rows.tex").write_text("\n".join(er) + "\n")

    # ------------------------------------------------------------ Table 4: post-training under shift
    pr = RES / "pooltest" / "report-68B.json"
    if not pr.exists():
        pr = RES / "pool" / "report-68B.json"
    if pr.exists():
        rep = json.loads(pr.read_text())
        names = [("Stage 1 (flow matching)", "Stage 1"), ("+ temperature scaling on the pool", "Stage 1 + TS (pool)"),
                 ("+ supervised fine-tuning on the pool", "SFT on pool"), ("+ RL, full network, KL anchor", "RL (Brier term only, beta 0.5)"),
                 ("+ RL calibrator (ours)", "RL calibrator (Brier term only)"), None,
                 ("\\quad calibrator + GRPO term", "RL calibrator (GRPO + Brier term)"),
                 ("\\quad calibrator, GRPO term only", "RL calibrator (GRPO only)"),
                 ("\\quad calibrator + GRPO, in-distribution pool", "RL calibrator, in-distribution pool")]
        lines = []
        sp = "test" if "pooltest" in str(pr) else "development"
        for item in names:
            if item is None:
                lines.append("\\midrule"); continue
            lab_, key = item
            if key not in rep:
                continue
            v = rep[key]
            n = v.get(f"transfer-v4.{sp}", {}); i = v.get(f"decision-v7.{sp}", {}); ph = v.get("pool.heldout", {})
            row = [fmt(n.get('acc')), fmt(n.get('nll')), fmt(n.get('ece')), fmt(n.get('confident_error_rate'), pct=True), fmt(n.get('aurc')),
                   fmt(i.get('acc')), fmt(i.get('nll')), fmt(i.get('confident_error_rate'), pct=True), fmt(ph.get('nll'))]
            if "(ours)" in lab_:
                lab_ = "\\textbf{" + lab_ + "}"
            lines.append(f"{lab_} & " + " & ".join(row) + "\\\\")
        (PAPER / "tables" / "rl_rows.tex").write_text("\n".join(lines) + "\n")
        keys = {"Stage 1": "St", "Stage 1 + TS (pool)": "Ts", "SFT on pool": "Sft", "RL (Brier term only, beta 0.5)": "Full",
                "RL calibrator (Brier term only)": "Cal", "RL calibrator (GRPO + Brier term)": "CalPg", "RL calibrator (GRPO only)": "PgOnly",
                "RL calibrator, in-distribution pool": "CalIn"}
        for name, k in keys.items():
            if name not in rep:
                continue
            v = rep[name]
            sp = "test" if "pooltest" in str(pr) else "development"
            for part, pk in ((f"transfer-v4.{sp}", "New"), (f"decision-v7.{sp}", "Id"), ("pool.heldout", "Pool")):
                e = v.get(part, {})
                for mm, mk in (("acc", "Acc"), ("nll", "Nll"), ("ece", "Ece"), ("confident_error_rate", "Cer"), ("aurc", "Aurc")):
                    if e.get(mm) is not None:
                        macro(f"Pool{k}{pk}{mk}", fmt(e[mm], 3, pct=(mm == "confident_error_rate"), lead=False))
            if "T" in v:
                macro(f"Pool{k}T", f"{v['T']:.2f}")

    # ------------------------------------------------------------ Table 5: ablations + NFE
    ab = RES / "ablations.json"
    if ab.exists():
        a = json.loads(ab.read_text())
        lines = []
        for name, v in a["ablations"].items():
            d7, t4 = v.get("decision-v7", {}), v.get("transfer-v4", {})
            lines.append(f"{name} & {fmt(d7.get('acc'))} & {fmt(d7.get('nll'))} & {fmt(t4.get('acc'))} & {fmt(t4.get('nll'))} & {fmt(t4.get('ece'))} & {fmt(t4.get('confident_error_rate'), pct=True)}\\\\")
        (PAPER / "tables" / "abl_rows.tex").write_text("\n".join(lines) + "\n")
        akeys = {"full model": "Ref", "cross-entropy classifier (no flow, no self-cond.)": "Ce", "no ConvNeXt V2 blocks": "NoCnx",
                 "no self-conditioning": "NoSc", "one-hot score encoding (no CDF)": "Onehot", "flow loss only (no cross-entropy)": "NoCe"}
        for name, v in a["ablations"].items():
            k = akeys.get(name)
            if not k:
                continue
            for suite, sk in (("decision-v7", "ID"), ("transfer-v4", "New")):
                e = v.get(suite, {})
                for mm, mk in (("acc", "Acc"), ("nll", "Nll"), ("brier", "Brier"), ("ece", "Ece"), ("confident_error_rate", "Cer")):
                    if e.get(mm) is not None:
                        macro(f"Abl{k}{sk}{mk}", fmt(e[mm], 3, pct=(mm == "confident_error_rate"), lead=False))
    for tag, k in (("ref", "Ref"), ("onehot", "Onehot")):
        f = RES / "abl" / tag / "decision-v7.development.report.json"
        if f.exists():
            c = json.loads(f.read_text())["clean"]
            macro(f"Abl{k}Rps", fmt(c["ranked_probability_score"], 3, lead=False)); macro(f"Abl{k}Mae", fmt(c["score_mae"], 3, lead=False))
        nl = []
        for nfe, v in a.get("nfe", {}).items():
            d7, t4 = v.get("decision-v7", {}), v.get("transfer-v4", {})
            nl.append(f"{nfe} & {fmt(d7.get('acc'))} & {fmt(d7.get('nll'))} & {fmt(t4.get('acc'))} & {fmt(t4.get('nll'))} & {fmt(t4.get('ece'))}\\\\")
        (PAPER / "tables" / "nfe_rows.tex").write_text("\n".join(nl) + "\n")

    # ------------------------------------------------------------ appendix: per-source accuracy on new sources
    tasks = ["mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq", "contrastive_authorization", "contrastive_deadline",
             "composition_final_combination", "composition_final_negation", "composition_final_exception"]
    who = have(["Rill-32M", "Rill-68M", "Rill-150M", "Kev-0.8B", "Kev-4B", "SemIf-Qwen3-0.6B", "SemIf-Qwen3.5-4B", "Layla-Mistral-7B", "NLI-DeBERTa-large"])
    lines = []
    for n in who:
        t = agg[n]["suites"].get("transfer-v4", {}).get("tasks", {})
        lines.append(f"{agg[n]['info']['label']} & " + " & ".join(fmt((t.get(k) or {}).get("acc"), 2) for k in tasks) + "\\\\")
    (PAPER / "tables" / "pertask_rows.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ appendix: raw vs temperature-scaled (T fit on decision-v7 calibration)
    who_ts = have(["Rill-32M", "Rill-68M", "Rill-150M", "Rill-68M-A", "Rill-68M-B", "Kev-0.8B", "Kev-4B", "SemIf-Qwen3-0.6B", "SemIf-MiniCPM5-2B",
                   "SemIf-Qwen3.5-4B", "Layla-TinyLlama-1.1B", "Layla-Qwen1.5-1.8B", "Layla-Phi2-2.7B", "Layla-Mistral-7B",
                   "NLI-DeBERTa-base", "NLI-BART-large", "NLI-DeBERTa-large"])
    lines = []
    for n in who_ts:
        e = agg[n]["suites"].get("transfer-v4", {})
        r, t = e.get("raw_unscaled") or e.get("raw") or {}, e.get("ts")
        if not t:
            continue
        lines.append(f"{agg[n]['info']['label']} & {fmt(e.get('T'), 2, lead=False)} & {fmt(r.get('nll'))} & {fmt(t.get('nll'))} & {fmt(r.get('ece'))} & {fmt(t.get('ece'))} & "
                     f"{fmt(r.get('confident_error_rate'), pct=True)} & {fmt(t.get('confident_error_rate'), pct=True)} & {fmt(r.get('aurc'))} & {fmt(t.get('aurc'))}\\\\")
    for n, k in (("Kev-0.8B", "KevSmall"), ("SemIf-Qwen3.5-4B", "SemIfFour"), ("Rill-68M", "RSixtyEight"), ("Rill-150M", "ROneFifty")):
        t = (agg.get(n, {}).get("suites", {}).get("transfer-v4", {}) or {}).get("ts")
        if t:
            macro(f"Ts{k}Ece", fmt(t["ece"], 3, lead=False)); macro(f"Ts{k}Cer", fmt(t["confident_error_rate"], pct=True))
    for n in have(["Rill-68M+RL", "Rill-150M+RL"]):
        r = agg[n]["suites"].get("transfer-v4", {}).get("raw") or {}
        lines.append(f"{agg[n]['info']['label']} & -- & {fmt(r.get('nll'))} & -- & {fmt(r.get('ece'))} & -- & {fmt(r.get('confident_error_rate'), pct=True)} & -- & {fmt(r.get('aurc'))} & --\\\\")
    (PAPER / "tables" / "ts_rows.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ appendix: accuracy with 95% record-clustered bootstrap intervals
    who_ci = have(["Rill-32M", "Rill-68M", "Rill-150M", "Kev-0.8B", "Kev-4B", "SemIf-Qwen3-0.6B", "SemIf-MiniCPM5-2B", "SemIf-Qwen3.5-4B",
                   "Layla-TinyLlama-1.1B", "Layla-Qwen1.5-1.8B", "Layla-Phi2-2.7B", "Layla-Mistral-7B", "NLI-DeBERTa-base", "NLI-BART-large",
                   "NLI-DeBERTa-large"])
    lines = []
    for n in who_ci:
        cells = []
        for suite in ("decision-v7", "transfer-v4", "transfer-v9"):
            e = agg[n]["suites"].get(suite, {})
            a_, ci_ = (e.get("raw") or {}).get("acc"), e.get("acc_ci")
            cells.append(f"{fmt(a_)} [{fmt(ci_[0])}, {fmt(ci_[1])}]" if (a_ is not None and ci_) else "--")
        lines.append(f"{agg[n]['info']['label']} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables" / "ci_rows.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ pareto data
    ddir = PAPER / "figures" / "data"; ddir.mkdir(parents=True, exist_ok=True)
    fams = {"rill": ["Rill-32M", "Rill-68M", "Rill-150M"], "kev": ["Kev-0.8B", "Kev-4B", "Kev-9B"],
            "semif": ["SemIf-Qwen3-0.6B", "SemIf-MiniCPM5-2B", "SemIf-Qwen3.5-4B"],
            "layla": ["Layla-TinyLlama-1.1B", "Layla-Qwen1.5-1.8B", "Layla-Phi2-2.7B", "Layla-Mistral-7B"],
            "nli": ["NLI-DeBERTa-base", "NLI-BART-large", "NLI-DeBERTa-large"]}
    accs = []
    for fam, names in fams.items():
        rows = ["params acc cer"]
        for n in names:
            e = agg.get(n, {}).get("suites", {}).get("transfer-v4", {}).get("raw", {})
            if e.get("acc") is not None and agg[n]["info"].get("params"):
                accs.append(e["acc"])
                rows.append(f"{agg[n]['info']['params']:.6g} {e['acc']:.4f} {100 * (e.get('confident_error_rate') or 0):.2f}")
        if len(rows) == 1:
            rows.append("nan nan nan")
        (ddir / f"pareto_{fam}.dat").write_text("\n".join(rows) + "\n")
    if accs:
        macro("ParetoYminAuto", f"{max(0.0, math.floor(min(accs) * 20) / 20 - 0.05):.2f}")
        macro("ParetoYmaxAuto", f"{min(1.0, math.ceil(max(accs) * 20) / 20 + 0.02):.2f}")

    # ------------------------------------------------------------ macros used in the text
    for o, key in (("Rill-32M", "RThirtyTwo"), ("Rill-68M", "RSixtyEight"), ("Rill-150M", "ROneFifty"), ("Rill-68M+RL", "RSixtyEightRL"),
                   ("Rill-32M+RL", "RThirtyTwoRL"), ("Rill-150M+RL", "ROneFiftyRL"), ("Rill-68M-A", "RSixtyEightA"), ("Rill-150M-A", "ROneFiftyA"),
                   ("Rill-68M-B", "RSixtyEightB"), ("Rill-68M+RL-canon", "RCanon"), ("Rill-68M-s1", "RSixtyEightSOne"), ("Rill-68M-s1+RL", "RSixtyEightSOneRL"), ("Kev-0.8B", "KevSmall"),
                   ("Kev-4B", "KevFour"), ("Kev-9B", "KevNine"), ("SemIf-Qwen3-0.6B", "SemIfSmall"), ("SemIf-MiniCPM5-2B", "SemIfMid"),
                   ("SemIf-Qwen3.5-4B", "SemIfFour"), ("Layla-TinyLlama-1.1B", "LaylaTiny"), ("Layla-Qwen1.5-1.8B", "LaylaQwen"),
                   ("Layla-Phi2-2.7B", "LaylaPhi"), ("Layla-Mistral-7B", "LaylaSeven"), ("NLI-DeBERTa-base", "NliBase"),
                   ("NLI-BART-large", "NliBart"), ("NLI-DeBERTa-large", "NliLarge")):
        if o not in agg:
            continue
        s9, s4 = agg[o]["suites"].get("transfer-v9", {}), agg[o]["suites"].get("transfer-v4", {})
        if (s9.get("unknowable") or {}).get("share_at_0_9") is not None:
            macro(f"{key}Unk", fmt(s9["unknowable"]["share_at_0_9"], pct=True))
        if (s4.get("variants") or {}).get("none_absent") is not None:
            macro(f"{key}Nab", fmt(s4["variants"]["none_absent"], 3, lead=False))
        if (s4.get("variants") or {}).get("none_present") is not None:
            macro(f"{key}Npr", fmt(s4["variants"]["none_present"], 3, lead=False))
        if s4.get("perm_flip") is not None:
            macro(f"{key}Flip", fmt(s4["perm_flip"], pct=True))
        for sname, sk in (("semif-v1", "SemIfSuite"), ("wanli-v1", "Wanli"), ("scienthoon-v1", "Tickets")):
            e = (agg[o]["suites"].get(sname, {}).get("raw") or {})
            if e.get("acc") is not None:
                macro(f"{key}{sk}Acc", fmt(e["acc"], 3, lead=False))
        for s, sk in (("decision-v7", "ID"), ("transfer-v4", "New"), ("transfer-v9", "Nine")):
            e = agg[o]["suites"].get(s, {}).get("raw", {}) or {}
            if e.get("confident_error_rate") is not None and e.get("n"):
                macro(f"{key}{sk}CerCount", f"{round(e['confident_error_rate'] * e['n']):d}")
                macro(f"{key}{sk}N", f"{e['n']:,}")
            for mm, mk in (("acc", "Acc"), ("nll", "Nll"), ("brier", "Brier"), ("ece", "Ece"), ("confident_error_rate", "Cer"), ("coverage_at_5pct_error", "Cov"),
                           ("mean_conf", "Conf")):
                if e.get(mm) is not None:
                    macro(f"{key}{sk}{mk}", fmt(e[mm], 3, pct=(mm == "confident_error_rate"), lead=False))
    for stem, key in (("cpu4-rill-68m-C", "LatRSixtyEight"), ("cpu4-rill-32m-C", "LatRThirtyTwo"), ("cpu4-rill-150m-C", "LatROneFifty"),
                      ("cpu4-kev-0.8b", "LatKevSmall"), ("cpu4-semif-qwen3-0.6b", "LatSemIfSmall"), ("cpu4-semif-qwen3.5-4b", "LatSemIfFour"),
                      ("cpu4-semif-minicpm5-2b", "LatSemIfMid")):
        if stem in lat:
            macro(key + "One", f"{lat[stem]['single']['median_ms']:,.0f}")
            macro(key + "Three", f"{lat[stem]['three']['median_ms']:,.0f}")
    for stem, key in (("cpu4-rill-32m-C-g8w8", "LatQThirtyTwo"), ("cpu4-rill-68m-C-g8w8", "LatQSixtyEight"), ("cpu4-rill-150m-C-g8w8", "LatQOneFifty")):
        if stem in lat:
            macro(key + "One", f"{lat[stem]['single']['median_ms']:,.0f}")
            macro(key + "Three", f"{lat[stem]['three']['median_ms']:,.0f}")
            macro(key + "MB", f"{lat[stem]['info']['model_mb']:.1f}")
    if "Rill-68M-A" in agg and "Rill-150M-A" in agg:
        g = agg["Rill-150M-A"]["suites"]["decision-v7"]["raw"]["acc"] - agg["Rill-68M-A"]["suites"]["decision-v7"]["raw"]["acc"]
        macro("ROneFiftyAGain", f"{100 * g:.1f}")
    for o, key in (("Kev-0.8B", "KevSmall"), ("Kev-4B", "KevFour")):
        for sname, sk in (("decision-v7", "ID"), ("transfer-v4", "New")):
            e = (agg.get(o, {}).get("suites", {}).get(sname, {}) or {}).get("raw_unscaled") or {}
            for mm, mk in (("nll", "Nll"), ("brier", "Brier"), ("ece", "Ece"), ("confident_error_rate", "Cer"), ("coverage_at_5pct_error", "Cov")):
                if e.get(mm) is not None:
                    macro(f"{key}Raw{sk}{mk}", fmt(e[mm], 3, pct=(mm == "confident_error_rate"), lead=False))
        e = agg.get(o, {}).get("suites", {}).get("transfer-v4", {})
        if e.get("shipped_T"):
            macro(f"{key}ShippedT", f"{e['shipped_T']:.2f}")
    rl = [agg[n]["suites"]["transfer-v4"]["raw"]["confident_error_rate"] for n in ("Rill-32M+RL", "Rill-68M+RL", "Rill-150M+RL") if n in agg]
    if rl:
        macro("RLCerMin", fmt(min(rl), pct=True)); macro("RLCerMax", fmt(max(rl), pct=True))
    if "Rill-68M" in agg and "Kev-0.8B" in agg:
        g = agg["Kev-0.8B"]["suites"]["decision-v7"]["raw"]["acc"] - agg["Rill-68M"]["suites"]["decision-v7"]["raw"]["acc"]
        macro("IdGapKev", f"{100 * g:.1f}")
    # accuracy on the in-distribution questions SemIf can represent (Banking77's 77-78-option questions excluded)
    import numpy as np
    from kev.metrics import scored_rows
    reg = json.loads((RES / "registry.json").read_text())
    ref = RES / "baselines/semif-qwen3.5-4b/decision-v7.test.rows.json"
    if ref.exists():
        unsup = {(r["id"], r["question"]) for r in json.loads(ref.read_text()) if r.get("unsupported")}
        for o, key in (("Rill-68M", "RSixtyEight"), ("Rill-150M", "ROneFifty"), ("Kev-0.8B", "KevSmall"), ("SemIf-Qwen3-0.6B", "SemIfSmall"),
                       ("SemIf-MiniCPM5-2B", "SemIfMid"), ("SemIf-Qwen3.5-4B", "SemIfFour"), ("Layla-Mistral-7B", "LaylaSeven")):
            f = ROOT / reg[o]["dir"] / "decision-v7.test.rows.json"
            if o in reg and f.exists():
                rows = [r for r in scored_rows(json.loads(f.read_text())) if (r["id"], r["question"]) not in unsup]
                macro(f"Sup{key}Acc", fmt(float(np.mean([int(np.argmax(r["p"]) == r["label"]) for r in rows])), 3, lead=False))
                if o == "Rill-68M":
                    macro("SupN", f"{len(rows):,}")
                    sup_rill = float(np.mean([int(np.argmax(r["p"]) == r["label"]) for r in rows]))
                if o == "SemIf-Qwen3.5-4B" and "sup_rill" in dir():
                    macro("SupGapSemIf", f"{100 * (sup_rill - float(np.mean([int(np.argmax(r['p']) == r['label']) for r in rows]))):.1f}")
    base = lat.get("cpu4-rill-68m-C")
    llm = [lat[k] for k in ("cpu4-kev-0.8b", "cpu4-semif-qwen3-0.6b", "cpu4-semif-minicpm5-2b", "cpu4-semif-qwen3.5-4b") if k in lat]
    if base and llm:
        r = [x["single"]["median_ms"] / base["single"]["median_ms"] for x in llm]
        macro("SpeedMin", f"{min(r):.0f}"); macro("SpeedMax", f"{max(r):.0f}")
        r3 = [x["three"]["median_ms"] / base["three"]["median_ms"] for x in llm]
        macro("SpeedThreeMin", f"{min(r3):.0f}"); macro("SpeedThreeMax", f"{max(r3):.0f}")
    w = lat.get("wasm-rill-32m-C")
    if w:
        for set_, k in (("single", "One"), ("three", "Three")):
            macro(f"WasmMs{k}", f"{w[set_]['wasm_median_ms']:,.0f}")
            macro(f"JsMs{k}", f"{w[set_]['js_median_ms']:,.0f}")
        macro("DemoMB", f"{w['model_mb']:.1f}")
    qc = sorted((ROOT / "demo" / "site_build").glob("quant_check.*.json"))
    if qc:
        agree = n = 0
        for f in qc:
            d = json.loads(f.read_text()); agree += d["g8w8"]["agreement"] * d["n_questions"]; n += d["n_questions"]
        macro("QuantAgree", f"{100 * agree / n:.1f}")
        macro("QuantN", f"{n:,}")
    (PAPER / "numbers_auto.tex").write_text("\n".join(MACROS) + "\n")
    print(f"table rows: main {len(order)}, halluc {len(order2)}, gen {len(gen_rows)}, eff {len(er)}; macros {len(MACROS)}")


if __name__ == "__main__":
    main()
